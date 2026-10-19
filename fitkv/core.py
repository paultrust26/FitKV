"""Core FitKV computations, following the method in the paper.

Notation (per layer l, per query window W of size w, per block j of P tokens):
  A[t, j]     block attention of window query t to block j (already averaged over heads)
  mu[j]       mean of A[:, j] over the window
  sigma[j]    sample standard deviation of A[:, j] over the window
  D_l         disagreement of layer l: mean(sigma) / (mean(mu) + eps)
  lambda      routing scalar in [0, 1]: min(1, mean_{l in late} D_l / 2)
  B_x         budget of an input x: floor(n - lambda * (n - B))
  s[j]        block score: mu[j] + lambda_l * sigma[j] + eta * g_tilde[j]
"""

import math

import torch
import torch.nn.functional as F

EPS = 1e-8
SMOOTH_WIDTH = 5


def fitkv_budget(n: int, lam: float, minimum: int) -> int:
    """Budget of an input with n tokens, routing scalar lam, and minimum budget."""
    if n <= minimum:
        return n
    budget = math.floor(n - lam * (n - minimum))
    return max(minimum, min(budget, n))


def late_layers(num_layers: int) -> list:
    """Indices of the last third of the layers (0-based): floor(2L/3)+1 .. L (1-based)."""
    start = (2 * num_layers) // 3  # 0-based index of layer floor(2L/3)+1
    return list(range(start, num_layers))


def block_attention(queries: torch.Tensor, keys: torch.Tensor, block: int, window: int) -> torch.Tensor:
    """Block attention of the last `window` queries to the context blocks.

    queries: [H, n, d] post rotary queries (one row per key value head)
    keys:    [H, n, d] cached keys
    Returns  [window, m] where m is the number of whole context blocks.
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
    """D_l from the block attention matrix A ([w, m])."""
    mu = A.mean(dim=0)                       # [m]
    sigma = A.std(dim=0, unbiased=True)      # [m]
    return float(sigma.mean() / (mu.mean() + EPS))


def routing_scalar(per_layer_D: list) -> float:
    """lambda = min(1, mean of late layer disagreements / 2)."""
    return min(1.0, sum(per_layer_D) / len(per_layer_D) / 2.0)


def layer_scores(A: torch.Tensor, value_term: torch.Tensor, lam_l: float, eta: float = 0.3) -> torch.Tensor:
    """Block score s[j] = mu[j] + lam_l * sigma[j] + eta * g_tilde[j]."""
    mu = A.mean(dim=0)
    sigma = A.std(dim=0, unbiased=True)
    return mu + lam_l * sigma + eta * value_term


def fitkv_select_blocks(scores: torch.Tensor, keep: int, always_keep: int = 2) -> torch.Tensor:
    """Indices of the blocks to keep: the last `always_keep` blocks plus the top scores."""
    m = scores.shape[0]
    if keep >= m:
        return torch.arange(m)
    protected = torch.arange(max(0, m - always_keep), m)
    scores = scores.clone()
    scores[protected] = float("inf")
    return torch.topk(scores, keep).indices.sort().values
