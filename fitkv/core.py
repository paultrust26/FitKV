"""Core FitKV computations, following the method in the paper.

Notation (per layer l, per query window W of size w, per block j of P tokens):
  A[t, j]     block attention of window query t to block j (already averaged over heads)
  mu[j]       mean of A[:, j] over the window
  sigma[j]    sample standard deviation of A[:, j] over the window
  D_l         disagreement of layer l: mean(sigma) / (mean(mu) + eps)
  lambda      routing scalar in [0, 1]: min(1, mean_{l in late} D_l / lambda_scale)
  B_x         budget of an input x: floor(n - lambda * (n - B))
  s[j]        block score: mu[j] + lambda_l * sigma[j] + eta * g_tilde[j]
"""

import math

import torch
import torch.nn.functional as F

from .config import get

EPS = get("numerics", "eps")
SMOOTH_WIDTH = get("numerics", "smooth_width")


def fitkv_budget(n: int, lam: float, minimum: int) -> int:
    """Budget of an input from its length, routing scalar and minimum budget.

    Args:
        n (int): prompt length in tokens (n >= 1).
        lam (float): routing scalar in [0, 1].
        minimum (int): budget floor B in tokens.

    Returns:
        int: the budget B_x, with minimum <= B_x <= n; equals n when n <= minimum.
    """
    if n <= minimum:
        return n
    budget = math.floor(n - lam * (n - minimum))
    return max(minimum, min(budget, n))


def late_layers(num_layers: int) -> list:
    """Indices of the late layers, the last third of the network (0-based).

    Args:
        num_layers (int): total number of layers L (>= 1).

    Returns:
        list[int]: indices floor(2L/3) .. L-1.
    """
    start = (2 * num_layers) // 3  # 0-based index of layer floor(2L/3)+1
    return list(range(start, num_layers))


def block_attention(queries: torch.Tensor, keys: torch.Tensor, block: int, window: int) -> torch.Tensor:
    """Window query attention pooled into context blocks, averaged over heads.

    Args:
        queries (Tensor [H, n, d]): post rotary queries per head.
        keys (Tensor [H, n, d]): cached keys per head.
        block (int): tokens per block P.
        window (int): observation window w.

    Returns:
        Tensor [w, m]: attention of each window query to each of the m = floor((n-w)/P) blocks.
    """
    H, n, d = keys.shape
    n_ctx = ((n - window) // block) * block
    m = n_ctx // block
    q = queries[:, -window:, :].float()                       # [H, w, d]
    k = keys[:, :n_ctx, :].float()                            # [H, n_ctx, d]
    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    attn = F.softmax(logits, dim=-1)                          # [H, w, n_ctx]
    # smoothing along the token axis (moving average, SnapKV convention)
    smooth = F.avg_pool1d(attn.reshape(H * window, 1, n_ctx), SMOOTH_WIDTH,
                          stride=1, padding=SMOOTH_WIDTH // 2)
    smooth = smooth.reshape(H, window, n_ctx)
    per_block = smooth.reshape(H, window, m, block).sum(dim=-1)  # [H, w, m]
    return per_block.mean(dim=0)                                # [w, m]


def disagreement(A: torch.Tensor) -> float:
    """Layer disagreement D = mean(sigma) / mean(mu) over the context blocks.

    Args:
        A (Tensor [w, m]): block attention from block_attention.

    Returns:
        float: the disagreement D_l of the layer.
    """
    mu = A.mean(dim=0)                       # [m]
    sigma = A.std(dim=0, unbiased=True)      # [m]
    return float(sigma.mean() / (mu.mean() + EPS))


def routing_scalar(per_layer_D: list) -> float:
    """Routing scalar lambda, the mean late layer disagreement halved and clipped to [0, 1].

    Args:
        per_layer_D (list[float]): disagreement of each late layer.

    Returns:
        float: lambda in [0, 1].
    """
    return min(1.0, sum(per_layer_D) / len(per_layer_D) / get("fitkv", "lambda_scale"))


def layer_scores(A: torch.Tensor, value_term: torch.Tensor, lam_l: float, eta: float = get("fitkv", "value_weight")) -> torch.Tensor:
    """Block scores mu + lam_l * sigma + eta * g for one layer.

    Args:
        A (Tensor [w, m]): block attention of the window.
        value_term (Tensor [m]): rescaled value term g_tilde.
        lam_l (float): layer weight lambda_l in [0, 1].
        eta (float): value term weight.

    Returns:
        Tensor [m]: one score per block.
    """
    mu = A.mean(dim=0)
    sigma = A.std(dim=0, unbiased=True)
    return mu + lam_l * sigma + eta * value_term


def fitkv_select_blocks(scores: torch.Tensor, keep: int, always_keep: int = get("fitkv", "always_keep_blocks")) -> torch.Tensor:
    """Indices of the highest scoring blocks, always including the last blocks.

    Args:
        scores (Tensor [m]): block scores.
        keep (int): number of blocks to keep.
        always_keep (int): number of final blocks forced into the selection.

    Returns:
        Tensor [keep] of int64: sorted block indices.
    """
    m = scores.shape[0]
    if keep >= m:
        return torch.arange(m)
    protected = torch.arange(max(0, m - always_keep), m)
    scores = scores.clone()
    scores[protected] = float("inf")
    return torch.topk(scores, keep).indices.sort().values
