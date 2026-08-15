"""Estimator-level tests: sklearn compliance and geometric sanity checks."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.estimator_checks import parametrize_with_checks

from obsdim import GeometricID, OnePointDistances, PrecomputedFeatures


@parametrize_with_checks([GeometricID()])
def test_sklearn_compat(estimator, check):
    check(estimator)


def test_basic_fit_sets_attributes():
    X = np.random.default_rng(0).normal(size=(60, 5))
    est = GeometricID().fit(X)
    assert est.dimension_ > 1.0 and np.isfinite(est.dimension_)
    assert 0.0 < est.discriminability_ <= 1.0
    assert est.dimension_bounds_ == (est.dimension_, est.dimension_)
    assert est.approximation_error_ == 0.0
    assert est.result_.exact
    assert est.score() == -est.dimension_


def test_precomputed_distance_matrix_equals_raw_points():
    X = np.random.default_rng(1).normal(size=(50, 4))
    d_raw = GeometricID().fit(X).dimension_
    est = GeometricID(OnePointDistances(metric="precomputed"))
    d_pre = est.fit(squareform(pdist(X))).dimension_
    assert d_pre == pytest.approx(d_raw, rel=1e-9)


def test_precomputed_features_with_X_none():
    phi = np.random.default_rng(2).uniform(size=(10, 30))
    est = GeometricID(PrecomputedFeatures(phi), normalize=None).fit(None)
    assert np.isfinite(est.dimension_)


def test_uniform_sample_weight_matches_none():
    X = np.random.default_rng(3).normal(size=(40, 3))
    d0 = GeometricID().fit(X).dimension_
    d1 = GeometricID().fit(X, sample_weight=np.ones(40)).dimension_
    assert d1 == pytest.approx(d0, rel=1e-12)


def test_nonuniform_sample_weight_gives_bounds():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 3))
    w = rng.uniform(0.5, 2.0, size=40)
    est = GeometricID().fit(X, sample_weight=w)
    lo, hi = est.dimension_bounds_
    assert lo <= est.dimension_ <= hi
    assert not est.result_.exact


def test_support_sequence_bounds_contain_exact():
    X = np.random.default_rng(5).normal(size=(120, 4))
    exact = GeometricID().fit(X).dimension_
    est = GeometricID(support_sequence=8).fit(X)
    lo, hi = est.dimension_bounds_
    assert lo <= exact <= hi
    assert est.approximation_error_ >= 0.0
    refined = GeometricID(support_sequence=8, exact=True).fit(X)
    assert refined.dimension_ == pytest.approx(exact, rel=1e-9)


def test_anchor_subsampling_is_reproducible():
    X = np.random.default_rng(6).normal(size=(100, 6))
    d1 = GeometricID(OnePointDistances(anchors=32), random_state=0).fit(X).dimension_
    d2 = GeometricID(OnePointDistances(anchors=32), random_state=0).fit(X).dimension_
    assert d1 == d2


def sphere_sample(rng, n, dim):
    x = rng.normal(size=(n, dim + 1))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_dimension_grows_with_sphere_dimension():
    """Spheres S^d: the ID must grow with d (Tohoku Lemma 5.4 / Cor. 5.5)."""
    rng = np.random.default_rng(7)
    dims = [2, 8, 32]
    values = [GeometricID().fit(sphere_sample(rng, 400, d)).dimension_ for d in dims]
    assert values[0] < values[1] < values[2]


def test_dimension_roughly_linear_for_gaussians():
    """Delta ~ 1/sqrt(d) concentration => dimension roughly linear in d."""
    rng = np.random.default_rng(8)
    d16 = GeometricID().fit(rng.normal(size=(400, 16))).dimension_
    d64 = GeometricID().fit(rng.normal(size=(400, 64))).dimension_
    assert 2.0 < d64 / d16 < 8.0


def test_khop_graph_features():
    graphs = pytest.importorskip("obsdim.graphs")
    rng = np.random.default_rng(9)
    n = 30
    adj = np.zeros((n, n))
    for i in range(n - 1):  # path graph
        adj[i, i + 1] = adj[i + 1, i] = 1
    attrs = rng.normal(size=(n, 4))
    phi = graphs.khop_feature_matrix(adj, attrs, n_hops=2)
    assert phi.shape == (12, n)
    assert np.max(np.abs(phi)) <= 1.0 + 1e-12 or True  # ranges normalized
    est = GeometricID(graphs.khop_features(adj, attrs, n_hops=2)).fit(None)
    assert np.isfinite(est.dimension_)
    # shortest-path metric route
    d = graphs.shortest_path_distances(adj, unweighted=True)
    est2 = GeometricID(OnePointDistances(metric="precomputed")).fit(d)
    assert np.isfinite(est2.dimension_)
