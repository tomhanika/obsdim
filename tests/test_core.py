"""Tests of the core functional against exact values derived from the papers.

Oracles used:

- Nominal scale (Tohoku Section 6.2.2): the geometric data set of the
  concept lattice of ``([n], [n], =)`` has dimension exactly ``n**4``.
- Contranominal scale (Tohoku Section 6.2.2): dimension tends to ``64/9``.
  (The closed formula printed in the paper uses the trapezoidal rule; the
  tests below use the exact step-function integral of Definition 4.1,
  ``Delta = (1/n) * sum_{k >= n/2+1} (k-1)/n``, which has the same limit.)
- Scaled indicator features (Tohoku Proposition 6.2): partial diameter of
  ``(c 1_B)_* nu`` is ``c`` iff ``alpha < nu(B) < 1 - alpha`` else 0.
- Brute-force subset enumeration validating TMLR Lemma 3.1/3.5.
- Support-sequence bounds and exact refinement (TMLR Section 4).
"""

import itertools
import math

import numpy as np
import pytest

from obsdim import (
    compute_dimension,
    discriminability,
    observable_diameter,
    observable_dimension,
    partial_diameter,
)


def nominal_phi(n):
    """Features of the concept lattice of the nominal scale ([n],[n],=).

    The non-trivial concepts are ({g},{g}); the associated features are
    ``nu_G({g}) * 1_{{g}} = (1/n) * 1_{{g}}`` (Tohoku Definition 6.1).
    """
    return np.eye(n) / n


def contranominal_phi(n):
    """Representative features of the contranominal scale ([n],[n],!=).

    Concepts are ``(A, [n] \\ A)`` for every ``A``; the feature of ``(A, B)``
    is ``(|A|/n) * 1_B``.  Since ``phi_{k,f}`` depends only on the multiset
    of feature values, one representative per ``|A| = a`` suffices
    (permutation invariance, TMLR Lemma 3.3).
    """
    idx = np.arange(n)
    return np.stack([(a / n) * (idx >= a) for a in range(1, n)])


def contranominal_delta_exact(n):
    """Exact Delta of the contranominal scale (step-function integral)."""
    return sum((k - 1) / n for k in range(2, n + 1) if 2 * k >= n + 2) / n


def test_nominal_scale_dimension_is_n_to_the_4():
    for n in [2, 3, 5, 8, 13]:
        dim = observable_dimension(nominal_phi(n), normalize=None)
        assert dim == pytest.approx(n**4, rel=1e-12)


def test_contranominal_scale_exact_small_n():
    for n in range(2, 13):
        delta = discriminability(contranominal_phi(n), normalize=None)
        assert delta == pytest.approx(contranominal_delta_exact(n), rel=1e-12)


def test_contranominal_scale_limit_64_over_9():
    dim = observable_dimension(contranominal_phi(500), normalize=None)
    assert dim == pytest.approx(64 / 9, abs=0.05)


def test_prop_6_2_weighted_indicator_partial_diameter():
    c = 0.7
    values = np.array([0.0, c])
    for p in [0.15, 0.3, 0.5, 0.62, 0.9]:
        weights = np.array([1 - p, p])
        for alpha in [0.05, 0.2, 0.35, 0.55, 0.8]:
            expected = c if alpha < p < 1 - alpha else 0.0
            got = partial_diameter(values, alpha, weights=weights)
            assert got == pytest.approx(expected, abs=1e-12), (p, alpha)


def brute_force_delta(phi):
    """TMLR Eq. (3) evaluated literally over all subsets of size k."""
    n = phi.shape[1]
    total = 0.0
    for k in range(2, n + 1):
        best = 0.0
        for f in range(phi.shape[0]):
            row = phi[f]
            smallest = min(
                max(row[list(m)]) - min(row[list(m)])
                for m in itertools.combinations(range(n), k)
            )
            best = max(best, smallest)
        total += min(best, 1.0)
    return total / n


def test_sliding_window_matches_subset_enumeration():
    rng = np.random.default_rng(42)
    phi = rng.normal(size=(3, 7))
    assert discriminability(phi) == pytest.approx(brute_force_delta(phi), rel=1e-12)


def test_delta_equals_integral_of_observable_diameter():
    """Theorem 3.2 consistency: Delta = int_0^1 ObsDiam(D; -alpha) da."""
    rng = np.random.default_rng(0)
    phi = rng.uniform(size=(4, 25))
    n = 25
    # ObsDiam is constant on each interval ((n-k)/n, (n-k+1)/n); evaluate at
    # midpoints, which exercises the independent single-alpha code path.
    integral = (
        sum(
            min(observable_diameter(phi, 1 - (k - 0.5) / n), 1.0)
            for k in range(1, n + 1)
        )
        / n
    )
    assert discriminability(phi) == pytest.approx(integral, rel=1e-12)


def test_observable_diameter_alpha_extremes():
    phi = np.array([[0.0, 1.0, 3.0]])
    # alpha ~ 0: the full support is needed -> full range
    assert observable_diameter(phi, 0.0) == pytest.approx(3.0)
    # alpha >= 1 - 1/n: a single point suffices -> 0
    assert observable_diameter(phi, 0.9) == 0.0


def test_support_sequence_bounds_and_refinement():
    rng = np.random.default_rng(7)
    phi = rng.normal(size=(5, 80))
    exact = compute_dimension(phi)
    approx = compute_dimension(phi, support_sequence=6)
    lo, hi = approx.delta_bounds
    assert lo <= exact.delta <= hi
    dlo, dhi = approx.dimension_bounds
    assert dlo <= exact.dimension <= dhi
    # TMLR Cor. 4.5: E(s, D) bounds the relative error of either endpoint.
    assert (dhi - exact.dimension) / exact.dimension <= approx.approximation_error
    assert not approx.exact

    refined = compute_dimension(phi, support_sequence=6, exact=True)
    assert refined.exact
    assert refined.delta == pytest.approx(exact.delta, rel=1e-12)
    assert refined.dimension == pytest.approx(exact.dimension, rel=1e-12)


def test_explicit_support_sequence_and_edges():
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(3, 20))
    exact = compute_dimension(phi)
    # A full support sequence has no gaps: bounds collapse to the exact value.
    full = compute_dimension(phi, support_sequence=list(range(2, 21)))
    assert full.delta_bounds[0] == pytest.approx(exact.delta, rel=1e-12)
    assert full.delta_bounds[1] == pytest.approx(exact.delta, rel=1e-12)
    with pytest.raises(ValueError):
        compute_dimension(phi, support_sequence=[2, 25])


def test_uniform_weights_route_to_exact_path():
    rng = np.random.default_rng(3)
    phi = rng.normal(size=(4, 15))
    res = compute_dimension(phi, weights=np.full(15, 2.0))
    assert res.exact
    assert res.dimension == pytest.approx(compute_dimension(phi).dimension)


def test_weighted_point_mass_equals_duplicated_point():
    """A point of double weight = the same point duplicated (mm-space view)."""
    rng = np.random.default_rng(5)
    base = rng.normal(size=(3, 10))
    dup = np.concatenate([base, base[:, :1]], axis=1)  # duplicate point 0
    weights = np.ones(10)
    weights[0] = 2.0
    res_w = compute_dimension(base, weights=weights, grid=11)
    res_dup = compute_dimension(dup)
    # all step positions lie on the grid -> the upper sum is exact
    assert res_w.delta_bounds[1] == pytest.approx(res_dup.delta, rel=1e-9)
    lo, hi = res_w.delta_bounds
    assert lo <= res_dup.delta <= hi + 1e-12


def test_weighted_grid_bounds_are_nested():
    rng = np.random.default_rng(11)
    phi = rng.normal(size=(4, 30))
    weights = rng.uniform(0.1, 1.0, size=30)
    coarse = compute_dimension(phi, weights=weights, grid=32)
    fine = compute_dimension(phi, weights=weights, grid=64)
    assert coarse.delta_bounds[0] <= fine.delta_bounds[0] + 1e-12
    assert fine.delta_bounds[1] <= coarse.delta_bounds[1] + 1e-12
    assert fine.delta_bounds[0] <= fine.delta_bounds[1]


def test_support_sequence_rejects_weighted():
    phi = np.ones((2, 5))
    with pytest.raises(ValueError):
        compute_dimension(phi, weights=[1, 2, 1, 1, 1], support_sequence=3)


def test_cap_wedge_one():
    phi = np.array([[0.0, 10.0]])
    capped = compute_dimension(phi, normalize=None)
    assert capped.delta == pytest.approx(0.5)  # min(10, 1) / 2
    assert capped.dimension == pytest.approx(4.0)
    uncapped = compute_dimension(phi, normalize=None, cap=None)
    assert uncapped.delta == pytest.approx(5.0)
    assert uncapped.dimension == pytest.approx(1 / 25)


def test_normalize_diameter_is_scale_invariant():
    rng = np.random.default_rng(2)
    phi = rng.normal(size=(4, 12))
    a = compute_dimension(phi, normalize="diameter")
    b = compute_dimension(1000.0 * phi, normalize="diameter")
    assert a.dimension == pytest.approx(b.dimension, rel=1e-12)
    assert a.scale == pytest.approx(1000.0 * b.scale, rel=1e-12)


def test_homogeneity_of_delta():
    """tau * ObsDiam(D; -a) = ObsDiam(tau * D; -a), Tohoku Section 6.1.1."""
    rng = np.random.default_rng(4)
    phi = rng.uniform(0, 0.1, size=(3, 9))
    d1 = discriminability(phi, cap=None)
    d3 = discriminability(3.0 * phi, cap=None)
    assert d3 == pytest.approx(3.0 * d1, rel=1e-12)


def test_constant_and_permuted_features_are_neutral():
    """TMLR Lemma 3.3."""
    rng = np.random.default_rng(6)
    phi = rng.normal(size=(3, 14))
    padded = np.concatenate([phi, np.full((1, 14), 7.0), phi[:1, ::-1]], axis=0)
    assert observable_dimension(padded) == pytest.approx(
        observable_dimension(phi), rel=1e-12
    )


def test_degenerate_inputs():
    assert observable_dimension(np.zeros((3, 1))) == math.inf  # single point
    assert observable_dimension(np.zeros((3, 8))) == math.inf  # constant features
    assert observable_dimension(np.zeros((0, 8))) == math.inf  # empty family
    with pytest.raises(ValueError):
        observable_dimension(np.zeros(5))
    with pytest.raises(ValueError):
        compute_dimension(np.ones((2, 4)), weights=[-1, 1, 1, 1])
    with pytest.raises(ValueError):
        compute_dimension(np.ones((2, 4)), weights=[1, 1])


def test_dimension_is_at_least_one():
    # Delta <= 1 by the wedge-1 cap, hence dimension >= 1 (Tohoku Prop. 5.3).
    rng = np.random.default_rng(8)
    phi = rng.normal(size=(5, 20)) * 50
    assert observable_dimension(phi, normalize=None) >= 1.0
