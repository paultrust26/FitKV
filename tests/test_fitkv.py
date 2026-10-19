import torch

from fitkv.core import fitkv_budget, late_layers, routing_scalar
from fitkv.evict import compress_kv_cache


def test_budget_limits():
    assert fitkv_budget(8000, 0.0, 128) == 8000   # full agreement keeps the whole cache
    assert fitkv_budget(8000, 1.0, 128) == 128    # full disagreement shrinks to the minimum
    assert fitkv_budget(8000, 0.1, 128) == 7212
    assert fitkv_budget(8000, 0.8, 128) == 1702


def test_budget_never_below_minimum_or_above_length():
    for lam in torch.linspace(0, 1, 11).tolist():
        b = fitkv_budget(1000, lam, 128)
        assert 128 <= b <= 1000


def test_late_layers_are_last_third():
    assert late_layers(28)[0] == 18 and late_layers(28)[-1] == 27
    assert len(late_layers(28)) == 10


def test_routing_scalar_clipped():
    assert routing_scalar([5.0, 5.0]) == 1.0
    assert routing_scalar([0.0, 0.0]) == 0.0


def test_compress_keeps_window_and_reduces_cache():
    torch.manual_seed(0)
    L, H, n, d = 6, 2, 4096, 16
    K = [torch.randn(H, n, d) for _ in range(L)]
    V = [torch.randn(H, n, d) for _ in range(L)]
    Q = [torch.randn(H, n, d) for _ in range(L)]
    budget, lam, kept = compress_kv_cache(K, V, Q, minimum=128)
    assert budget < n
    for k, t in zip(K, kept):
        assert k.shape[1] == t.numel() < n
        assert t[-64:].tolist() == list(range(n - 64, n))


def test_short_input_is_untouched():
    torch.manual_seed(0)
    K = [torch.randn(2, 100, 8) for _ in range(4)]
    V = [torch.randn(2, 100, 8) for _ in range(4)]
    Q = [torch.randn(2, 100, 8) for _ in range(4)]
    budget, _, kept = compress_kv_cache(K, V, Q, minimum=128)
    assert kept is None and budget == 100 and K[0].shape[1] == 100
