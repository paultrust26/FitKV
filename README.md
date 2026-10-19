# FitKV

Training free KV cache compression that sets a **per input** budget.

FitKV reads how much the final prompt queries disagree about which parts of the
context to attend to, in the late layers of the model. Inputs whose queries agree
keep almost their full cache; inputs whose queries disagree shrink toward a
minimum budget that you set. Blocks are then selected per layer with the same
attention statistics and a value term.

## How it works

For each late layer (the last third of the layers):

1. The last `w` prompt queries attend to the context, pooled into blocks of `P` tokens.
2. Each block gets a mean `mu` and a standard deviation `sigma` over the queries.
3. The layer's disagreement is `D = mean(sigma) / mean(mu)`.

The routing scalar is `lambda = min(1, mean(D) / 2)`, and the budget of the input is

    B_x = floor(n - lambda * (n - B))

where `n` is the prompt length and `B` is the minimum budget. Every layer then keeps the
`floor(B_x / P)` highest scoring blocks, always including the observation window and the last two blocks.

The block score is `mu + lambda_l * sigma + eta * g`, where `g` is a value term that
measures how far a block's mean value lies from the window's attention output.

## Install

```bash
pip install -r requirements.txt
```

## Use

```python
from fitkv import compress_kv_cache

# keys, values, queries: one [H, n, d] tensor per layer, after the prompt pass
budget, lam, kept = compress_kv_cache(keys, values, queries, minimum=128)
```

`keys` and `values` are replaced in place with the compressed tensors. `budget` is the
per input budget `B_x` and `lam` is the routing scalar.

A complete example with a Hugging Face model:

```bash
python examples/compress_prompt.py --model Qwen/Qwen2.5-0.5B-Instruct --budget 128
```

## Tests

```bash
python -m pytest tests/
```

## Parameters

| Name | Default | Meaning |
|------|---------|---------|
| `minimum` (B) | required | Minimum budget in tokens per layer |
| `block` (P) | 16 | Tokens per block |
| `window` (w) | 64 | Observation window size in queries |
| `eta` | 0.3 | Weight of the value term |

The scale of `1/2` in `lambda` was fixed by hand and not tuned.

## Scope

This repository contains the method. It does not include the benchmark harness used
for the paper's experiments (LongBench runs, baselines and result aggregation).

## Citation

```bibtex
@inproceedings{fitkv2026,
  title={FitKV: Fitting the KV Cache Budget to Each Input for Long Context Inference},
  author={Paul Trust},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
