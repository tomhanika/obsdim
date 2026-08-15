"""Tests for family-layer normalization semantics.

The documented rules (module docstring of ``obsdim._families``):

- global scaling = change of units, same geometric data set (Tohoku §6.1.1);
- per-column scaling (``CoordinateProjections(scale="range")``) = the
  sup-metric on min-max-scaled columns — a *different*, precisely
  identified geometric data set, invariant under per-column affine maps;
- per-member scaling in unions makes heterogeneous members commensurable
  while preserving each member's internal relative scales.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from obsdim import (
    CoordinateProjections,
    FeatureScaler,
    OnePointDistances,
    PrecomputedFeatures,
    UnionFamily,
    observable_dimension,
)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    # columns with wildly different scales ("mixed units")
    return rng.normal(size=(50, 4)) * np.array([1.0, 100.0, 0.01, 5.0])


def minmax(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def test_projections_induce_sup_metric_on_scaled_columns(X):
    """d_F(x, y) = max_j |phi_j(x) - phi_j(y)| equals the Chebyshev metric
    of the min-max-scaled data — the documented semantics of scale="range".
    """
    phi = CoordinateProjections(scale="range").evaluate(X)
    d_induced = np.max(np.abs(phi[:, :, None] - phi[:, None, :]), axis=0)
    d_chebyshev = cdist(minmax(X), minmax(X), "chebyshev")
    np.testing.assert_allclose(d_induced, d_chebyshev, atol=1e-12)


def test_projections_range_scale_is_affine_invariant(X):
    """Per-column min-max scaling kills per-column units entirely."""
    X2 = X * np.array([3.0, 0.5, 1000.0, 2.0]) + np.array([1.0, -7.0, 0.0, 42.0])
    a = CoordinateProjections(scale="range").evaluate(X)
    b = CoordinateProjections(scale="range").evaluate(X2)
    # identical up to the per-column offset, hence identical dimension
    np.testing.assert_allclose(
        a - a.min(axis=1, keepdims=True), b - b.min(axis=1, keepdims=True), atol=1e-12
    )
    assert observable_dimension(a) == pytest.approx(observable_dimension(b), rel=1e-9)


def test_projections_global_scale_keeps_relative_column_scales(X):
    """scale="global" is one common unit (TMLR Def. 5.1 / ID4GeoL d_max)."""
    phi = CoordinateProjections(scale="global").evaluate(X)
    d_max = np.max(X.max(axis=0) - X.min(axis=0))
    np.testing.assert_allclose(phi, X.T / d_max, atol=1e-12)
    # a global rescaling of the data does not change the features
    np.testing.assert_allclose(
        phi, CoordinateProjections(scale="global").evaluate(10.0 * X), atol=1e-12
    )


def test_projections_raw_and_validation(X):
    np.testing.assert_allclose(CoordinateProjections(scale=None).evaluate(X), X.T)
    with pytest.raises(ValueError):
        CoordinateProjections(scale="bogus").evaluate(X)
    with pytest.raises(ValueError):
        CoordinateProjections().evaluate(X[:, 0])


def test_projections_range_scale_noop_on_scaled_data(X):
    """On already min-max-scaled data, per-column scaling changes nothing."""
    Xs = minmax(X)
    np.testing.assert_allclose(
        CoordinateProjections(scale="range").evaluate(Xs),
        CoordinateProjections(scale=None).evaluate(Xs),
        atol=1e-12,
    )


def test_feature_scaler_global_equals_functional_normalize(X):
    """FeatureScaler(per_feature=False) == normalize="diameter" (§6.1.1)."""
    fam = OnePointDistances()
    scaled = FeatureScaler(fam, per_feature=False).evaluate(X)
    dim_family_layer = observable_dimension(scaled, normalize=None)
    dim_functional = observable_dimension(fam.evaluate(X), normalize="diameter")
    assert dim_family_layer == pytest.approx(dim_functional, rel=1e-12)


def test_feature_scaler_per_feature_unit_ranges(X):
    phi = FeatureScaler(OnePointDistances(), per_feature=True).evaluate(X)
    ranges = phi.max(axis=1) - phi.min(axis=1)
    np.testing.assert_allclose(ranges, 1.0, atol=1e-12)


def test_feature_scaler_changes_the_geometry(X):
    """Per-feature scaling of distance functions is NOT a change of units."""
    raw = observable_dimension(OnePointDistances().evaluate(X), normalize="diameter")
    per_feature = observable_dimension(
        FeatureScaler(OnePointDistances(), per_feature=True).evaluate(X)
    )
    assert per_feature != pytest.approx(raw, rel=1e-6)


def test_feature_scaler_passes_requires_data_through():
    pre = PrecomputedFeatures(np.random.default_rng(1).uniform(size=(3, 8)))
    assert not FeatureScaler(pre).requires_data
    assert FeatureScaler(OnePointDistances()).requires_data


def test_constant_features_survive_scaling():
    phi = np.vstack([np.linspace(0, 1, 10), np.full(10, 3.0)])
    scaled = FeatureScaler(PrecomputedFeatures(phi)).evaluate()
    assert np.all(np.isfinite(scaled))
    np.testing.assert_allclose(scaled[1], phi[1])  # constant row untouched
    assert observable_dimension(scaled) == pytest.approx(
        observable_dimension(phi[:1]), rel=1e-12
    )


def test_union_normalize_members_is_codomain_scale_invariant():
    rng = np.random.default_rng(2)
    phi_a = rng.uniform(size=(4, 25))
    phi_b = rng.uniform(size=(3, 25))
    for c in [1.0, 0.01, 1e4]:
        union = UnionFamily(
            [PrecomputedFeatures(phi_a), PrecomputedFeatures(c * phi_b)],
            normalize_members=True,
        )
        dim = observable_dimension(union.evaluate(None), normalize=None)
        if c == 1.0:
            reference = dim
    assert dim == pytest.approx(reference, rel=1e-12)


def test_union_normalize_members_equals_manual_wrapping(X):
    members = [OnePointDistances(), CoordinateProjections(scale=None)]
    auto = UnionFamily(members, normalize_members=True).evaluate(X)
    manual = UnionFamily(
        [FeatureScaler(m, per_feature=False) for m in members]
    ).evaluate(X)
    np.testing.assert_allclose(auto, manual, atol=1e-12)


def test_union_default_remains_raw(X):
    members = [OnePointDistances(), CoordinateProjections(scale=None)]
    raw = UnionFamily(members).evaluate(X)
    blocks = [m.evaluate(X) for m in members]
    np.testing.assert_allclose(raw, np.concatenate(blocks, axis=0), atol=1e-12)
