import itertools
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pytest
import torch

from ebsl.core import (
    assignment_edges,
    compression_ratio,
    fit_baseline,
    fit_er,
    reconstruct,
    solve_assignment,
    spatial_edges,
)
from ebsl.data import stratified_split
from ebsl.supervised import er_proxy, nearest_neighbor, nullspace_projection, vector_features


def test_neighborhood_edge_count_matches_paper():
    rows, cols = spatial_edges(64, 64)
    assert len(rows) == 98596
    assert np.all(np.abs(rows // 64 - cols // 64) <= 2)
    assert np.all(np.abs(rows % 64 - cols % 64) <= 2)


def test_assignment_matches_exhaustive_optimum_and_handles_zero():
    cost = np.random.default_rng(4).random((6, 6))
    np.fill_diagonal(cost, 0)
    cost[0, 0] = 3
    rows, cols = np.indices(cost.shape)
    gather = solve_assignment(6, rows.ravel(), cols.ravel(), cost.ravel())
    found = cost[gather, np.arange(6)].sum()
    expected = min(cost[list(p), np.arange(6)].sum() for p in itertools.permutations(range(6)))
    assert abs(found - expected) < 1e-12


@pytest.mark.parametrize("neighborhood", ["ln", "nn"])
def test_edges_match_direct_squared_distances(neighborhood):
    torch.manual_seed(3)
    x = torch.randn(9, 5, 5, dtype=torch.float64)
    b = torch.randn_like(x)
    p, q, cost = assignment_edges(x, b, neighborhood)
    expected = ((x.flatten(1)[:, p] - b.flatten(1)[:, q]) ** 2).sum(0).numpy()
    np.testing.assert_allclose(cost, expected, rtol=1e-12, atol=1e-12)
    assert set(np.arange(25)) <= set(p[p == q])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["glram", "2dpca"])
@pytest.mark.parametrize("neighborhood", ["ln", "nn"])
def test_er_monotone_bijective_and_reconstruction(device, mode, neighborhood):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(8)
    x = torch.randn(12, 6, 6, device=device, dtype=torch.float64)
    model = fit_er(x, (2, 2), mode, neighborhood, max_iter=8)
    assert np.all(np.diff([h["sse"] for h in model.history]) <= 1e-8)
    assert sorted(model.perm.tolist()) == list(range(36))
    expected = (
        reconstruct(model.rearrange(x), model.u, model.v)
        .flatten(1)[:, np.argsort(model.perm)]
        .reshape_as(x)
    )
    torch.testing.assert_close(model.reconstruct(x), expected)
    torch.testing.assert_close(
        model.u.T @ model.u, torch.eye(model.u.shape[1], device=device, dtype=x.dtype)
    )
    before = model.perm.copy()
    model.transform(torch.randn_like(x))
    np.testing.assert_array_equal(before, model.perm)


def test_full_rank_loss_zero_and_shared_permutation_invariant_distance():
    x = torch.randn(7, 5, 5, dtype=torch.float64)
    model = fit_baseline(x, (5, 5))
    torch.testing.assert_close(x, model.reconstruct(x))
    model.perm = np.random.default_rng(3).permutation(25)
    torch.testing.assert_close(
        torch.cdist(x.flatten(1), x.flatten(1)),
        torch.cdist(model.rearrange(x).flatten(1), model.rearrange(x).flatten(1)),
    )


def test_supervised_nullspace_removes_within_class_variation():
    torch.manual_seed(9)
    x = torch.randn(15, 6, 6, dtype=torch.float64)
    y = np.repeat(np.arange(5), 3)
    projected, info = nullspace_projection(x, y)
    assert info["within_rank"] == 10
    assert info["null_within_relative"] < 1e-12
    assert er_proxy(x, y, "lda", projected=projected).shape == (5, 6, 6)
    assert er_proxy(x, y, "mfa", k2=4, projected=projected).shape == (20, 6, 6)


def test_split_disjoint_complete_and_repeatable():
    y = np.repeat(np.arange(4), 10)
    a, b = stratified_split(y, 4, 10)
    assert not set(a) & set(b)
    assert sorted(np.r_[a, b].tolist()) == list(range(40))
    np.testing.assert_array_equal(np.bincount(y[a]), [4] * 4)
    np.testing.assert_array_equal(a, stratified_split(y, 4, 10)[0])


def test_cr_counts_permutation_overhead():
    identity = np.arange(4096)
    cr, bits = compression_ratio(100, (64, 64), (5, 5), "glram", identity)
    assert bits == 8192
    assert cr == 100 * 4096 / (100 * 25 + 128 * 5 + 8192 / 32)


def test_vector_baselines_support_strict_cuda_determinism():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(10)
        x = torch.randn(15, 6, 6, device="cuda")
        y = np.repeat(np.arange(5), 3)
        for kind in ["pca", "lda", "mfa"]:
            a, b, rank = vector_features(x, x[:3], y, kind, k2=5)
            assert a.shape == (15, rank) and b.shape == (3, rank)
            assert torch.isfinite(a).all()
            assert (nearest_neighbor(a, y, b) == y[:3]).all()
    finally:
        torch.use_deterministic_algorithms(previous)
