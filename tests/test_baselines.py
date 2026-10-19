import torch

from fitkv.baselines import (
    adakv_scores,
    h2o_scores,
    pyramidkv_layer_budget,
    quest_scores,
    select_blocks,
    snapkv_scores,
)
from fitkv.compare import METHODS, apply_method


def _cache(L=4, H=2, n=640, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: [torch.randn(H, n, d, generator=g) for _ in range(L)]
    return mk(), mk(), mk()


def test_select_blocks_keeps_last_two_and_top_scores():
    idx = select_blocks(torch.arange(10.0), keep=4)
    assert idx.tolist() == [6, 7, 8, 9]
    idx = select_blocks(torch.tensor([9.0, 0, 0, 0, 0, 0, 0, 0]), keep=3)
    assert 0 in idx.tolist() and 7 in idx.tolist() and 6 in idx.tolist()


def test_quest_bound_is_upper_bound_of_block_dot_products():
    torch.manual_seed(0)
    q = torch.randn(1, 8)          # one head, one query vector
    k = torch.randn(1, 64, 8)      # one head, 64 keys
    s = quest_scores(q, k, block=16)
    assert s.shape[0] == 4
    for b in range(4):
        exact = (k[0, b * 16:(b + 1) * 16] @ q[0]).max()
        assert s[b] >= exact - 1e-5


def test_scorers_return_one_score_per_block():
    torch.manual_seed(0)
    q = torch.randn(2, 640, 16)
    k = torch.randn(2, 640, 16)
    m = (640 - 64) // 16
    assert snapkv_scores(q, k, 16).shape[0] == m
    assert h2o_scores(q, k, 16).shape[0] == 640 // 16
    assert adakv_scores(q, k, 16).shape[0] == m


def test_pyramidkv_budgets_average_to_budget():
    L, B = 8, 256
    per = [pyramidkv_layer_budget(l, L, B, 16) for l in range(L)]
    assert abs(sum(per) / L - B) < 1
    assert per[0] > per[-1]


def test_every_method_runs_and_reports_budget():
    for method in METHODS:
        K, V, Q = _cache()
        budget = apply_method(method, K, V, Q, minimum=256)
        assert budget <= K[0].shape[1] if method != "full" else True
        assert K[0].shape == V[0].shape


def test_full_method_keeps_everything():
    K, V, Q = _cache()
    n = K[0].shape[1]
    apply_method("full", K, V, Q, minimum=256)
    assert K[0].shape[1] == n
