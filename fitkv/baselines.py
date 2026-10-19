"""Baseline page (block) scoring methods compared with FitKV in the paper.

Each scorer takes a prefill query window and the cached keys, and returns one
score per block. Blocks are then kept with `select_blocks`, which always keeps
the observation window and the last two blocks, as in the paper.

  quest_scores       Quest: upper bound of q.k per block, from block min and max keys
  snapkv_scores      SnapKV: attention of the final window, pooled over the window
  h2o_scores         H2O: attention accumulated over all prompt queries, sink blocks kept
  pyramidkv_budget   PyramidKV: per layer budget that decreases linearly with depth
  adakv_scores       Ada-KV: per head SnapKV scores with a global cross head competition
                     and a per head floor (block level adaptation of AdaKV)
"""

import math

import torch
import torch.nn.functional as F

from .config import get

SMOOTH_WIDTH = get("numerics", "smooth_width")


def select_blocks(scores: torch.Tensor, keep: int, always_keep: int = get("fitkv", "always_keep_blocks")) -> torch.Tensor:
    """Indices of the top scoring blocks, always including the last blocks.

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
    scores = scores.clone()
    scores[max(0, m - always_keep):] = float("inf")
    return torch.topk(scores, keep).indices.sort().values


def _block_mean(x: torch.Tensor, block: int) -> torch.Tensor:
    """Sum attention within consecutive blocks along the last dimension.

    Args:
        x (Tensor [..., n_ctx]): attention over context tokens.
        block (int): tokens per block.

    Returns:
        Tensor [..., m]: per block sums, with m = n_ctx / block.
    """
    *lead, n = x.shape
    return x.reshape(*lead, n // block, block).sum(dim=-1)


def quest_scores(q_last: torch.Tensor, keys: torch.Tensor, block: int) -> torch.Tensor:
    """Quest block scores: upper bound of query and key dot products per block.

    Args:
        q_last (Tensor [H, d]): last query of each head.
        keys (Tensor [H, n, d]): cached keys.
        block (int): tokens per block.

    Returns:
        Tensor [m]: one upper bound score per block, summed over heads.
    """
    H, n, d = keys.shape
    m = n // block
    kb = keys[:, : m * block, :].reshape(H, m, block, d)
    k_min, k_max = kb.min(dim=2).values, kb.max(dim=2).values      # [H, m, d]
    q = q_last.unsqueeze(1).expand(-1, m, -1)                       # [H, m, d]
    upper = torch.where(q >= 0, q * k_max, q * k_min).sum(dim=-1)   # [H, m]
    return upper.sum(dim=0)


def snapkv_scores(queries: torch.Tensor, keys: torch.Tensor, block: int, window: int = get("fitkv", "window")) -> torch.Tensor:
    """SnapKV block scores from the attention of the final window, pooled over the window.

    Args:
        queries (Tensor [H, n, d]): post rotary queries.
        keys (Tensor [H, n, d]): cached keys.
        block (int): tokens per block.
        window (int): observation window w.

    Returns:
        Tensor [m]: one score per block, averaged over heads.
    """
    H, n, d = keys.shape
    window = min(window, n - 1)
    n_ctx = ((n - window) // block) * block
    q = queries[:, -window:, :].float()
    logits = torch.matmul(q, keys[:, :n_ctx, :].float().transpose(-2, -1)) / math.sqrt(d)
    attn = F.softmax(logits, dim=-1).mean(dim=1)                    # [H, n_ctx]
    attn = F.avg_pool1d(attn.unsqueeze(1), SMOOTH_WIDTH, stride=1, padding=SMOOTH_WIDTH // 2).squeeze(1)
    return _block_mean(attn, block).mean(dim=0)                     # [m]


def h2o_scores(queries: torch.Tensor, keys: torch.Tensor, block: int, n_sink: int = get("baselines", "h2o_sink_tokens"), chunk: int = get("baselines", "h2o_chunk")) -> torch.Tensor:
    """H2O block scores from attention accumulated over all prompt queries.

    Args:
        queries (Tensor [H, n, d]): post rotary queries.
        keys (Tensor [H, n, d]): cached keys.
        block (int): tokens per block.
        n_sink (int): initial tokens always kept.
        chunk (int): query rows per chunk.

    Returns:
        Tensor [m]: accumulated attention per block; sink blocks receive the maximum score.
    """
    H, n, d = keys.shape
    m = n // block
    usable = m * block
    k = keys[:, :usable, :].float()
    acc = torch.zeros(H, usable)
    k_pos = torch.arange(usable)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        logits = torch.matmul(queries[:, s:e, :].float(), k.transpose(-2, -1)) / math.sqrt(d)
        mask = torch.arange(s, e).unsqueeze(1) >= k_pos.unsqueeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))
        acc += torch.nan_to_num(F.softmax(logits, dim=-1), nan=0.0).sum(dim=1)
    scores = acc.reshape(H, m, block).sum(dim=-1).mean(dim=0)       # [m]
    sink = max(1, (n_sink + block - 1) // block)
    scores[:sink] = scores.max() + 1.0
    return scores


def pyramidkv_layer_budget(layer: int, num_layers: int, budget: int, block: int,
                           window_blocks: int = get("baselines", "pyramidkv_window_blocks"), beta: int = get("baselines", "pyramidkv_beta")) -> int:
    """Per layer token budget from PyramidKV's linear schedule (layer 0 largest).

    Args:
        layer (int): 0-based layer index.
        num_layers (int): total layers L.
        budget (int): average budget B per layer.
        block (int): tokens per block.
        window_blocks (int): blocks kept as window.
        beta (int): smallest budget is the average divided by beta.

    Returns:
        int: the token budget of this layer, including the window.
    """
    window = window_blocks * block
    base = budget - window
    min_b = max(block, base // beta)
    max_b = base * 2 - min_b
    if num_layers == 1:
        return budget
    steps = (max_b - min_b) / (num_layers - 1)
    return max(min_b, round(max_b - layer * steps)) + window


def adakv_scores(queries: torch.Tensor, keys: torch.Tensor, block: int, window: int = get("fitkv", "window"),
                 floor_alpha: float = get("baselines", "adakv_floor_alpha")) -> torch.Tensor:
    """Ada-KV block scores with per head floors and a global competition across heads.

    Args:
        queries (Tensor [H, n, d]): post rotary queries.
        keys (Tensor [H, n, d]): cached keys.
        block (int): tokens per block.
        window (int): observation window w.
        floor_alpha (float): share of blocks each head keeps.

    Returns:
        Tensor [m]: one score per block, taken as the maximum over heads.
    """
    H, n, d = keys.shape
    window = min(window, n - 1)
    n_ctx = ((n - window) // block) * block
    m = n_ctx // block
    q = queries[:, -window:, :].float()
    logits = torch.matmul(q, keys[:, :n_ctx, :].float().transpose(-2, -1)) / math.sqrt(d)
    attn = F.softmax(logits, dim=-1)
    attn = F.avg_pool1d(attn.reshape(H * window, 1, n_ctx), SMOOTH_WIDTH,
                        stride=1, padding=SMOOTH_WIDTH // 2).reshape(H, window, n_ctx)
    per_head = _block_mean(attn.mean(dim=1), block)                  # [H, m]
    floor = max(1, math.ceil(floor_alpha * m))
    sentinel = (per_head.amax() * 100.0 + 1.0).reshape(())
    safe = per_head.clone()
    top = per_head.topk(floor, dim=-1).indices
    safe.scatter_(-1, top, sentinel.expand_as(top).to(safe.dtype))
    return safe.max(dim=0).values                                    # [m]
