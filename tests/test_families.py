"""Tests of the feature-family layer and its invariances."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist, squareform

from obsdim import (
    CustomFeatures,
    OnePointDistances,
    PrecomputedFeatures,
    UnionFamily,
    observable_dimension,
)


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(size=(40, 3))


def test_precomputed_metric_matches_euclidean(X):
    """The theory lives on mm-spaces: a distance matrix must be equivalent."""
    phi_raw = OnePointDistances().evaluate(X)
    phi_pre = OnePointDistances(metric="precomputed").evaluate(squareform(pdist(X)))
    np.testing.assert_allclose(phi_raw, phi_pre, atol=1e-12)


def test_string_metric_uses_cdist(X):
    phi = OnePointDistances(metric="cityblock").evaluate(X)
    np.testing.assert_allclose(phi, cdist(X, X, "cityblock"))


def test_callable_metric(X):
    phi = OnePointDistances(metric=lambda a, b: cdist(a, b, "chebyshev")).evaluate(X)
    np.testing.assert_allclose(phi, cdist(X, X, "chebyshev"))


def test_anchor_sampling(X):
    fam = OnePointDistances(anchors=7)
    phi = fam.evaluate(X, rng=123)
    assert phi.shape == (7, 40)
    np.testing.assert_allclose(phi, fam.evaluate(X, rng=123))  # reproducible
    # anchors=n is the exact family again
    np.testing.assert_allclose(
        OnePointDistances(anchors=40).evaluate(X), OnePointDistances().evaluate(X)
    )
    with pytest.raises(ValueError):
        OnePointDistances(anchors=41).evaluate(X)


def test_anchor_rows_are_rows_of_full_matrix(X):
    full = OnePointDistances().evaluate(X)
    sub = OnePointDistances(anchors=5).evaluate(X, rng=0)
    # every sampled feature is an exact one-point distance function
    matches = (np.abs(sub[:, None, :] - full[None, :, :]) < 1e-12).all(axis=2)
    assert matches.any(axis=1).all()


def test_precomputed_features_identity():
    phi = np.random.default_rng(1).uniform(size=(6, 20))
    fam = PrecomputedFeatures(phi)
    assert not fam.requires_data
    np.testing.assert_allclose(fam.evaluate(None), phi)
    np.testing.assert_allclose(
        PrecomputedFeatures(phi, lipschitz=2.0).evaluate(), phi / 2
    )


def test_custom_features_declared_vs_estimated_lipschitz():
    # On 1-D data the empirical constant of f(x) = 2x is exactly 2.
    X = np.linspace(0, 1, 30).reshape(-1, 1)
    funcs = [lambda X: 2.0 * X[:, 0]]
    declared = CustomFeatures(funcs, lipschitz=2.0).evaluate(X)
    estimated = CustomFeatures(funcs, lipschitz="estimate").evaluate(X, rng=0)
    np.testing.assert_allclose(declared, estimated, atol=1e-12)
    np.testing.assert_allclose(declared, X[:, 0][None, :], atol=1e-12)
    raw = CustomFeatures(funcs, lipschitz=None).evaluate(X)
    np.testing.assert_allclose(raw, 2.0 * X[:, 0][None, :], atol=1e-12)


def test_custom_features_per_feature_bounds(X):
    funcs = [lambda X: X[:, 0], lambda X: 3.0 * X[:, 1]]
    phi = CustomFeatures(funcs, lipschitz=[1.0, 3.0]).evaluate(X)
    np.testing.assert_allclose(phi[1], X[:, 1], atol=1e-12)


def test_union_family_stacks_and_is_antitone(X):
    dist = OnePointDistances()
    proj = CustomFeatures([lambda X: X[:, 0]], lipschitz=1.0)
    union = UnionFamily([dist, proj])
    phi = union.evaluate(X)
    assert phi.shape == (41, 40)
    # more features discriminate at least as well: dimension can only drop
    # (axiom of feature antitonicity; compare with identical normalization)
    dim_union = observable_dimension(phi, normalize=None)
    assert dim_union <= observable_dimension(dist.evaluate(X), normalize=None) + 1e-9
    assert dim_union <= observable_dimension(proj.evaluate(X), normalize=None) + 1e-9


def test_union_requires_data_propagates():
    pre = PrecomputedFeatures(np.ones((2, 5)))
    assert not UnionFamily([pre]).requires_data
    assert UnionFamily([pre, OnePointDistances()]).requires_data
