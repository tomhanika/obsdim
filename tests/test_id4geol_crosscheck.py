"""Cross-validation against ID4GeoL, the authors' experiment code.

The functions ``phi_fj`` / ``obs_diam`` and the ``min_Delta``/``max_Delta``
bound formulas below are transcribed verbatim (modulo renaming) from
https://github.com/mstubbemann/ID4GeoL (``approximate.py`` /
``error_experiment.py``, MIT-style research code).  obsdim must reproduce
their numbers exactly: same sliding-window ``phi``, same support-sequence
bounds, same dimension values.
"""

import numpy as np
import pytest

from obsdim import PrecomputedFeatures, compute_dimension

# --- reference implementation, transcribed from ID4GeoL ---------------------


def phi_fj(lf, j):
    return np.min(lf[j - 1 :] - lf[: 1 - j])


def obs_diam(X):
    return np.max(np.max(X, axis=0) - np.min(X, axis=0))


def id4geol_support_bounds(X, samples):
    """Support-sequence Delta bounds as in ID4GeoL's approximate.py."""
    n = X.shape[0]
    S = np.array(samples)
    gaps = S[1:] - S[:-1]
    lfs = [np.sort(X[:, i]) for i in range(X.shape[1])]
    max_phis = np.array([max(phi_fj(lf, j) for lf in lfs) for j in samples])
    min_delta = (np.sum(max_phis[:-1] * gaps) + max_phis[-1]) * (1 / n)
    max_delta = (np.sum(max_phis[1:] * gaps) + max_phis[0]) * (1 / n)
    return min_delta, max_delta


# --- cross-checks -----------------------------------------------------------


@pytest.fixture
def X():
    rng = np.random.default_rng(13)
    X = rng.standard_normal((200, 10))
    return X / obs_diam(X)  # ID4GeoL's normalization: coordinate range <= 1


def id4geol_samples(n, num):
    """ID4GeoL's support-sequence construction (densest near k = n)."""
    samples = sorted({int(x) for x in (n + 2 - np.geomspace(n, 2, num))})
    if samples[0] != 2:
        samples = [2, *samples]
    if samples[-1] != n:
        samples.append(n)
    return samples


def test_exact_delta_matches_id4geol(X):
    """Full support sequence: their bounds collapse onto the exact Delta."""
    n = X.shape[0]
    lo, hi = id4geol_support_bounds(X, list(range(2, n + 1)))
    assert lo == pytest.approx(hi, rel=1e-12)
    # ID4GeoL's features are the coordinate projections, i.e. Phi = X.T
    res = compute_dimension(X.T, normalize=None)
    assert res.delta == pytest.approx(lo, rel=1e-10)


def test_support_bounds_match_id4geol(X):
    n = X.shape[0]
    samples = id4geol_samples(n, 20)
    lo_ref, hi_ref = id4geol_support_bounds(X, samples)
    res = compute_dimension(X.T, normalize=None, support_sequence=samples)
    assert res.delta_bounds[0] == pytest.approx(lo_ref, rel=1e-10)
    assert res.delta_bounds[1] == pytest.approx(hi_ref, rel=1e-10)
    # and the derived dimension bounds / error, as in their scripts
    assert res.dimension_bounds[0] == pytest.approx(1 / hi_ref**2, rel=1e-10)
    assert res.dimension_bounds[1] == pytest.approx(1 / lo_ref**2, rel=1e-10)
    assert res.approximation_error == pytest.approx(
        (1 / lo_ref**2 - 1 / hi_ref**2) / (1 / hi_ref**2), rel=1e-10
    )


def test_precomputed_family_route_matches_id4geol(X):
    lo, hi = id4geol_support_bounds(X, list(range(2, X.shape[0] + 1)))
    phi = PrecomputedFeatures(X.T).evaluate()
    res = compute_dimension(phi, normalize=None)
    assert res.dimension == pytest.approx(1 / lo**2, rel=1e-9)
