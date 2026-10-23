"""Compress the KV cache of one prompt with FitKV.

Requires: torch and a Hugging Face causal LM. Run from the repository root:

    python examples/compress_prompt.py --model Qwen/Qwen2.5-0.5B-Instruct --budget 128
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitkv import compress_kv_cache  # noqa: E402


def capture_queries(model):
    """Record post rotary queries of every layer during the prompt pass.

    Query heads are averaged within each key value group (grouped query attention),
    as in the paper, so each layer gives one [kv_heads, n, d] tensor that matches its keys.
    """
    store = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def hook(module, args, kwargs, output, i=i):
            hs = kwargs["hidden_states"]
            pos_cos, pos_sin = kwargs["position_embeddings"]
            H = model.config.num_attention_heads
            d = model.config.hidden_size // H
            q = module.q_proj(hs).view(hs.shape[0], -1, H, d).transpose(1, 2)
            cos, sin = pos_cos.unsqueeze(1), pos_sin.unsqueeze(1)
            half = d // 2
            rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            q = (q * cos + rot * sin)[0]
            kv = model.config.num_key_value_heads
            store[i] = q.view(kv, H // kv, -1, d).mean(dim=1).detach()
        handles.append(attn.register_forward_hook(hook, with_kwargs=True))
    return store, handles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--budget", type=int, default=128, help="minimum budget B in tokens per layer")
    p.add_argument("--text", default="The capital of France is Paris. " * 200 + "Question: What is the capital of France?")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).eval()
    ids = tok(args.text, return_tensors="pt").input_ids
    queries, handles = capture_queries(model)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    for h in handles:
        h.remove()

    cache = out.past_key_values
    keys = [cache.layers[i].keys[0] for i in range(len(cache.layers))]
    values = [cache.layers[i].values[0] for i in range(len(cache.layers))]
    qs = [queries[i] for i in range(len(cache.layers))]
    budget, lam, _ = compress_kv_cache(keys, values, qs, minimum=args.budget)
    print(f"prompt tokens: {ids.shape[1]}  routing lambda: {lam:.3f}  budget: {budget}")


if __name__ == "__main__":
    main()
