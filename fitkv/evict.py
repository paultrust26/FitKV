"""Per input FitKV eviction for a per layer KV cache (prefill only)."""

import math

import torch
import torch.nn.functional as F

from .core import (
    fitkv_budget,
    fitkv_select_blocks,
    late_layers,
    routing_scalar,
)

SMOOTH_WIDTH = 5


def _head_block_attention(q, k, block, window):
    """Block attention per head. q, k: [H, n, d]. Returns [H, w, m] and n_ctx."""
    H, n, d = k.shape
    n_ctx = ((n - window) // block) * block
    qw = q[:, -window:, :].float()
    logits = torch.matmul(qw, k[:, :n_ctx, :].float().transpose(-2, -1)) / math.sqrt(d)
    attn = F.softmax(logits, dim=-1)                                 # [H, w, n_ctx]
    smooth = F.avg_pool1d(attn.reshape(H * window, 1, n_ctx), SMOOTH_WIDTH,
                          stride=1, padding=SMOOTH_WIDTH // 2).reshape(H, window, n_ctx)
    return smooth.reshape(H, window, n_ctx // block, block).sum(dim=-1), n_ctx


def _layer_disagreement(A_heads):
    """D_l (Eq. 4) from head averaged block attention [w, m]."""
    A = A_heads.mean(dim=0)
    mu = A.mean(dim=0)
    sigma = A.std(dim=0, unbiased=True)
    return float(sigma.mean() / (mu.mean() + 1e-8)), A, mu, sigma


def compress_kv_cache(keys_per_layer, values_per_layer, queries_per_layer, minimum,
                      block=16, window=64, eta=0.3):
    """Compress one prompt's KV cache in place with FitKV.

    keys_per_layer, values_per_layer: list over layers of [H, n, d] tensors
    queries_per_layer:                list over layers of [H, n, d] post rotary queries
    minimum:                          budget floor B in tokens per layer
    Returns (budget, lam, kept_indices_per_layer).
    """
    L = len(keys_per_layer)
    n = keys_per_layer[0].shape[1]
    if n <= minimum + window:
        return n, 0.0, None

    # 1. routing scalar from the late layers
    late = late_layers(L)
    D = []
    for l in late:
        A_heads, _ = _head_block_attention(queries_per_layer[l], keys_per_layer[l], block, window)
        D.append(_layer_disagreement(A_heads)[0])
    lam = routing_scalar(D)

    # 2. budget of this input
    budget = fitkv_budget(n, lam, minimum)
    if budget >= n:
        return n, lam, None
    keep_blocks = budget // block

    # 3. block selection in every layer
    kept = []
    for l in range(L):
        k, v, q = keys_per_layer[l], values_per_layer[l], queries_per_layer[l]
        A_heads, n_ctx = _head_block_attention(q, k, block, window)            # [H, w, m]
        d_l, A, mu, sigma = _layer_disagreement(A_heads)
        lam_l = min(1.0, d_l / 2.0)
        # value term (Eq. 8): distance of each block's mean value from the window output
        a_bar = A_heads.mean(dim=1)                                           # [H, m]
        v_blocks = v[:, :n_ctx, :].float().reshape(v.shape[0], -1, block, v.shape[-1]).mean(dim=2)
        out = (A_heads.mean(dim=1).unsqueeze(-1) * v_blocks).sum(dim=1) / (A_heads.mean(dim=1).sum(dim=1, keepdim=True) + 1e-8)
        dist = (v_blocks - out.unsqueeze(1)).norm(dim=-1) / math.sqrt(v.shape[-1])
        g = (a_bar * dist).mean(dim=0)                                        # [m]
        # rescale (Eq. 9) to the range of the attention terms
        g_t = g * ((mu + lam_l * sigma).max() / (g.max() + 1e-12))
        scores = mu + lam_l * sigma + eta * g_t                               # Eq. 10
        idx = fitkv_select_blocks(scores, keep_blocks, always_keep=2)
        tokens = torch.cat([torch.arange(int(b) * block, (int(b) + 1) * block) for b in idx]
                           + [torch.arange(n_ctx, n)]).to(k.device)
        keys_per_layer[l] = k[:, tokens, :]
        values_per_layer[l] = v[:, tokens, :]
        kept.append(tokens)
    return budget, lam, kept
