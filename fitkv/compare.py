"""Apply FitKV or a baseline to a prefill cache with one common interface.

Every method keeps blocks of P tokens per layer, always keeps the observation window
and the last two blocks, and uses the same budget floor, so differences come from the
selection rule.
"""

import math

import torch

from .baselines import (
    adakv_scores,
    h2o_scores,
    pyramidkv_layer_budget,
    quest_scores,
    select_blocks,
    snapkv_scores,
)
from .core import layer_scores
from .evict import _head_block_attention, _layer_disagreement, compress_kv_cache

METHODS = ("full", "fitkv", "fitkv_fixed", "snapkv", "h2o", "quest", "pyramidkv", "adakv")


def _keep_tokens(idx, block, n, n_ctx, device):
    return torch.cat([torch.arange(int(b) * block, (int(b) + 1) * block) for b in idx]
                     + [torch.arange(n_ctx, n)]).to(device)


def apply_method(method, keys, values, queries, minimum, block=16, window=64):
    """Compress keys and values (lists over layers of [H, n, d]) in place with `method`.

    Returns the per input budget used (int) for every method.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    L = len(keys)
    n = keys[0].shape[1]
    if method == "full" or n <= minimum + window:
        return n
    if method == "fitkv":
        budget, _, _ = compress_kv_cache(keys, values, queries, minimum, block, window)
        return budget

    n_ctx = ((n - window) // block) * block
    budget = minimum
    keep_blocks = budget // block

    for l in range(L):
        k, v, q = keys[l], values[l], queries[l]
        layer_keep = keep_blocks
        if method == "pyramidkv":
            layer_keep = pyramidkv_layer_budget(l, L, budget, block) // block
        if method == "quest":
            scores = quest_scores(q[:, -1], k, block)
        elif method == "snapkv":
            scores = snapkv_scores(q, k, block, window)
        elif method == "h2o":
            scores = h2o_scores(q, k, block)
        elif method == "pyramidkv":
            scores = snapkv_scores(q, k, block, window)
        elif method == "adakv":
            scores = adakv_scores(q, k, block, window)
        else:  # fitkv_fixed: FitKV selection with the fixed budget (lambda = 1)
            A_heads, _ = _head_block_attention(q, k, block, window)
            _, A, mu, sigma = _layer_disagreement(A_heads)
            lam_l = 1.0
            a_bar = A_heads.mean(dim=1)
            v_blocks = v[:, :n_ctx, :].float().reshape(v.shape[0], -1, block, v.shape[-1]).mean(dim=2)
            out = (a_bar.unsqueeze(-1) * v_blocks).sum(dim=1) / (a_bar.sum(dim=1, keepdim=True) + 1e-8)
            dist = (v_blocks - out.unsqueeze(1)).norm(dim=-1) / math.sqrt(v.shape[-1])
            g = (a_bar * dist).mean(dim=0)
            g_t = g * ((mu + lam_l * sigma).max() / (g.max() + 1e-12))
            scores = layer_scores(A, g_t, lam_l, 0.3)
        idx = select_blocks(scores, layer_keep, always_keep=2)
        tokens = _keep_tokens(idx, block, n, n_ctx, k.device)
        keys[l] = k[:, tokens, :]
        values[l] = v[:, tokens, :]
    return budget
