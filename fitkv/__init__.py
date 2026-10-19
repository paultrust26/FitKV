"""FitKV: input adaptive KV cache budgets from late layer attention disagreement."""

from .core import fitkv_budget, fitkv_select_blocks, late_layers, routing_scalar
from .evict import compress_kv_cache

__all__ = ["fitkv_budget", "fitkv_select_blocks", "late_layers", "routing_scalar", "compress_kv_cache"]
__version__ = "0.1.0"
