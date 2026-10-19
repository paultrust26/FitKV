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

SMOOTH_WIDTH = 5


def select_blocks(scores: torch.Tensor, keep: int, always_keep: int = 2) -> torch.Tensor:
    """Top `keep` blocks by score, with the last `always_keep` blocks forced in."""
    m = scores.shape[0]
    if keep >= m:
        return torch.arange(m)
    scores = scores.clone()
    scores[max(0, m - always_keep):] = float("inf")
    return torch.topk(scores, keep).indices.sort().values


def _block_mean(x: torch.Tensor, block: int) -> torch.Tensor:
    """Sum attention per block along the last dimension (x: [..., n_ctx])."""
    *lead, n = x.shape
    return x.reshape(*lead, n // block, block).sum(dim=-1)


def quest_scores(q_last: torch.Tensor, keys: torch.Tensor, block: int) -> torch.Tensor:
    """q_last: [H, d] last query; keys: [H, n, d]. Returns [m] block scores."""
    H, n, d = keys.shape
    m = n // block
    kb = keys[:, : m * block, :].reshape(H, m, block, d)
    k_min, k_max = kb.min(dim=2).values, kb.max(dim=2).values      # [H, m, d]
    q = q_last.unsqueeze(1).expand(-1, m, -1)                       # [H, m, d]
    upper = torch.where(q >= 0, q * k_max, q * k_min).sum(dim=-1)   # [H, m]
    return upper.sum(dim=0)


def snapkv_scores(queries: torch.Tensor, keys: torch.Tensor, block: int, window: int = 64) -> torch.Tensor:
    """queries, keys: [H, n, d]. Returns [m] block scores (head averaged)."""
    H, n, d = keys.shape
    window = min(window, n - 1)
    n_ctx = ((n - window) // block) * block
    q = queries[:, -window:, :].float()
    logits = torch.matmul(q, keys[:, :n_ctx, :].float().transpose(-2, -1)) / math.sqrt(d)
    attn = F.softmax(logits, dim=-1).mean(dim=1)                    # [H, n_ctx]
    attn = F.avg_pool1d(attn.unsqueeze(1), SMOOTH_WIDTH, stride=1, padding=SMOOTH_WIDTH // 2).squeeze(1)
    return _block_mean(attn, block).mean(dim=0)                     # [m]


def h2o_scores(queries: torch.Tensor, keys: torch.Tensor, block: int, n_sink: int = 4, chunk: int = 512) -> torch.Tensor:
    """Accumulated causal attention from all prompt queries. Returns [m] block scores."""
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
                           window_blocks: int = 4, beta: int = 20) -> int:
    """Per layer budget from PyramidKV's linear schedule (layer 0 largest)."""
    window = window_blocks * block
    base = budget - window
    min_b = max(block, base // beta)
    max_b = base * 2 - min_b
    if num_layers == 1:
        return budget
    steps = (max_b - min_b) / (num_layers - 1)
    return max(min_b, round(max_b - layer * steps)) + window


def adakv_scores(queries: torch.Tensor, keys: torch.Tensor, block: int, window: int = 64,
                 floor_alpha: float = 0.20) -> torch.Tensor:
    """Block level Ada-KV. queries, keys: [H, n, d]. Returns [m] block scores.

    Each head keeps its own SnapKV block scores. A block is kept if any head
    ranks it in its per head floor (the top `floor_alpha` share of blocks),
    or if it ranks highly in the global ordering across heads.
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
