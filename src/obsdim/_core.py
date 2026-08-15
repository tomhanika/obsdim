"""The dimension functional: from feature values to intrinsic dimension.

This module is the numerical heart of :mod:`obsdim`.  It implements, for a
finite geometric data set :math:`\\mathcal{D} = (X, F, \\mu)` represented by
the value matrix ``phi`` with ``phi[j, i] = f_j(x_i)`` and point weights
``weights`` (the measure :math:`\\mu`), the quantities defined in

- Hanika, Schneider, Stumme, *Intrinsic dimension of geometric data sets*,
  Tohoku Math. J. 74 (2022) 23-52 ("Tohoku paper"), and
- Stubbemann, Hanika, Schneider, *Intrinsic Dimension for Large-Scale
  Geometric Learning*, TMLR 2023 ("TMLR paper").

Correspondence between code and papers:

``partial_diameter``
    :math:`\\mathrm{PartDiam}(\\nu, 1-\\alpha) = \\inf\\{\\mathrm{diam}(B)
    \\mid B \\subseteq \\mathbb{R}\\ \\mathrm{Borel},\\ \\nu(B) \\ge
    1-\\alpha\\}` — Tohoku Definition 4.1.  For the empirical measure this is
    computed on contiguous windows of the sorted values (TMLR Lemma 3.1).
``observable_diameter``
    :math:`\\mathrm{ObsDiam}(\\mathcal{D}; -\\alpha) = \\sup_{f \\in F}
    \\mathrm{PartDiam}(f_*(\\mu), 1-\\alpha)` — Tohoku Definition 4.1.
``discriminability``
    :math:`\\Delta(\\mathcal{D}) = \\int_0^1 \\mathrm{ObsDiam}(\\mathcal{D};
    -\\alpha) \\wedge 1 \\; d\\alpha` — Tohoku Proposition 4.5 (the "wedge 1"
    cap is part of the definition there; TMLR Eq. (1) omits it because
    features are normalized).  For the uniform empirical measure this equals
    the exact finite formula :math:`\\Delta(\\mathcal{D}) = \\frac{1}{|X|}
    \\sum_{k=2}^{|X|} \\max_{f} \\min_{j} (l^{f}_{k+j} - l^{f}_{1+j})` of
    TMLR Theorem 3.2, evaluated with the sliding-window rule of TMLR
    Lemma 3.5 (Algorithm 1).
``observable_dimension`` / ``compute_dimension``
    :math:`\\partial(\\mathcal{D}) = 1 / \\Delta(\\mathcal{D})^2` — Tohoku
    Proposition 5.3, TMLR Eq. (1).  This is a dimension function in the sense
    of the axioms of Tohoku Definition 5.1.

Scalable approximation (TMLR Section 4): a *support sequence*
:math:`s = (2 = s_1 < \\dots < s_l = |X|)` yields lower/upper bounds
:math:`\\Delta_{s,-} \\le \\Delta \\le \\Delta_{s,+}` (TMLR Definition 4.2,
Corollary 4.3) and hence two-sided bounds on the dimension with certified
relative error :math:`E(s, \\mathcal{D})` (TMLR Definition 4.4).  With
``exact=True`` the bounds are refined to the exact value while skipping
redundant features (TMLR Lemma 4.7, Algorithm 2).

Everything is agnostic to where the features came from: the functional never
sees raw data or a metric, only ``(phi, weights)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._compat import as_float_array, get_namespace

__all__ = [
    "DimensionResult",
    "compute_dimension",
    "discriminability",
    "observable_diameter",
    "observable_dimension",
    "partial_diameter",
]

# Absolute tolerance on cumulative masses (weighted path only), guarding the
# ">= 1 - alpha" comparisons against floating-point rounding of weight sums.
_MASS_EPS = 1e-12


@dataclass(frozen=True)
class DimensionResult:
    """Full result of a dimension computation.

    Attributes
    ----------
    dimension : float
        The intrinsic dimension :math:`\\partial(\\mathcal{D}) =
        1/\\Delta(\\mathcal{D})^2` (Tohoku Prop. 5.3).  ``inf`` iff
        :math:`\\Delta = 0` (e.g. a single point, cf. Tohoku Remark 5.2).
        If the computation was approximate, this is the point estimate
        derived from the midpoint of ``delta_bounds``.
    dimension_bounds : tuple of float
        Certified interval :math:`[\\partial_{s,-}, \\partial_{s,+}]`
        containing the exact dimension (TMLR Def. 4.2 / Cor. 4.3).  Equal to
        ``(dimension, dimension)`` for exact computations.
    delta : float
        The discriminability :math:`\\Delta(\\mathcal{D}) \\in [0, 1]`
        (Tohoku Prop. 4.5).
    delta_bounds : tuple of float
        Certified interval :math:`[\\Delta_{s,-}, \\Delta_{s,+}]`.
    approximation_error : float
        The relative error bound :math:`E(s, \\mathcal{D}) =
        (\\partial_{s,+} - \\partial_{s,-}) / \\partial_{s,-}` (TMLR
        Def. 4.4); ``0.0`` for exact computations.
    n_points, n_features : int
        Size of the value matrix ``phi``.
    scale : float
        The factor :math:`\\tau` that was applied to the features
        (``normalize``; Tohoku Section 6.1.1).  ``1.0`` means the raw
        features were used.
    exact : bool
        Whether ``dimension`` is exact (up to floating point).
    """

    dimension: float
    dimension_bounds: tuple[float, float]
    delta: float
    delta_bounds: tuple[float, float]
    approximation_error: float
    n_points: int
    n_features: int
    scale: float
    exact: bool


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _validate_phi(phi, xp):
    phi = as_float_array(phi, xp)
    if phi.ndim != 2:
        raise ValueError(
            f"phi must be a 2-D array of shape (n_features, n_points), "
            f"got shape {phi.shape}"
        )
    return phi


def _validate_weights(weights, n, xp):
    """Normalize point weights to a probability vector."""
    w = as_float_array(weights, xp)
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    if bool(xp.any(w < 0)):
        raise ValueError("weights must be non-negative")
    total = float(xp.sum(w))
    if total <= 0:
        raise ValueError("weights must not sum to zero")
    return w / total


def _is_uniform(weights, xp):
    return bool(xp.all(weights == weights[0]))


def _resolve_scale(phi, normalize, xp):
    """The normalization factor tau of Tohoku Section 6.1.1.

    For a geometric data set, ``tau * ObsDiam(D; -a) = ObsDiam(tau * D; -a)``
    where ``tau * D`` rescales every feature.  ``normalize="diameter"`` uses
    ``tau = 1/diam(X, d_F)``; since ``d_F(x, y) = sup_f |f(x) - f(y)|``, the
    diameter equals the largest value range over all features.
    """
    if normalize is None or normalize is False:
        return 1.0
    if normalize == "diameter":
        if phi.shape[1] == 0:
            return 1.0
        ranges = xp.max(phi, axis=1) - xp.min(phi, axis=1)
        diam = float(xp.max(ranges)) if phi.shape[0] else 0.0
        return 1.0 / diam if diam > 0 else 1.0
    diam = float(normalize)
    if diam <= 0:
        raise ValueError("normalize must be 'diameter', None, or a positive number")
    return 1.0 / diam


def _cap_value(cap):
    if cap is None:
        return math.inf
    cap = float(cap)
    if cap <= 0:
        raise ValueError("cap must be positive or None")
    return cap


def _phi_window(sorted_phi, k, xp):
    """``phi_{k,f}`` for all features at once — TMLR Lemma 3.5.

    ``phi_{k,f} = min_j ( l^f_{k+j} - l^f_{1+j} )``, the smallest range of a
    window of ``k`` consecutive sorted feature values; equals
    ``PartDiam(f_* nu, k/n)`` for the uniform empirical measure by TMLR
    Lemma 3.1.  Returns an array of shape ``(n_features,)``.
    """
    n = sorted_phi.shape[1]
    if k <= 1:
        return xp.zeros(sorted_phi.shape[0], dtype=sorted_phi.dtype)
    return xp.min(sorted_phi[:, k - 1 :] - sorted_phi[:, : n - k + 1], axis=1)


def _make_support_sequence(support_sequence, n):
    """Validate/build a support sequence ``s = (2 = s_1 < ... < s_l = n)``.

    An integer ``l`` requests ``l`` geometrically spaced support points,
    dense near ``k = n`` (``s = n + 2 - geomspace(n, 2, l)``) — the choice
    used in TMLR Section 5.1 / ID4GeoL, since both ``phi_k`` and the gap
    contributions are largest for large ``k``.
    """
    if n < 2:
        return [n] if n else []
    if isinstance(support_sequence, (int, np.integer)):
        length = int(support_sequence)
        if length < 2:
            raise ValueError("an integer support_sequence must be >= 2")
        s = np.unique((n + 2 - np.geomspace(n, 2, num=length)).astype(int))
    else:
        s = np.unique(np.asarray(support_sequence, dtype=int))
        if s.size and (s[0] < 2 or s[-1] > n):
            raise ValueError(
                f"support sequence entries must lie in [2, {n}], got [{s[0]}, {s[-1]}]"
            )
    s = s.tolist()
    if not s or s[0] != 2:
        s.insert(0, 2)
    if s[-1] != n:
        s.append(n)
    return s


# ---------------------------------------------------------------------------
# Partial and observable diameters (single level alpha)
# ---------------------------------------------------------------------------


def _weighted_partdiams_at_masses(values, cum, masses, xp):
    """Partial diameters of one weighted feature at several mass levels.

    ``values`` are the sorted feature values, ``cum`` the padded cumulative
    weights (``cum[r]`` = total weight of the first ``r`` points, ``cum[-1] =
    1``).  For each mass level ``t``, returns the minimal diameter of a
    window of consecutive sorted values with total weight ``>= t`` — the
    weighted-measure analogue of TMLR Lemma 3.1 (the optimal Borel set of
    Tohoku Definition 4.1 can always be taken to be such a window).
    """
    n = values.shape[0]
    masses = xp.asarray(masses, dtype=values.dtype)
    n_levels = masses.shape[0]
    # r[i, t] = smallest r with cum[r] >= cum[i] + t  (window i .. r-1)
    targets = xp.reshape(cum[:n], (n, 1)) + xp.reshape(masses, (1, n_levels))
    r = xp.searchsorted(cum, xp.reshape(targets - _MASS_EPS, (-1,)))
    r = xp.reshape(r, (n, n_levels))
    valid = r <= n
    r_end = xp.clip(r - 1, 0, n - 1)
    diams = xp.take(values, xp.reshape(r_end, (-1,)))
    diams = xp.reshape(diams, (n, n_levels)) - xp.reshape(values, (n, 1))
    diams = xp.where(valid, diams, xp.asarray(math.inf, dtype=values.dtype))
    out = xp.min(diams, axis=0)
    # mass level <= 0 is satisfied by the empty set: PartDiam = 0
    return xp.where(masses <= _MASS_EPS, xp.zeros_like(out), out)


def _sort_with_weights(row, weights, xp):
    order = xp.argsort(row)
    values = xp.take(row, order)
    w = xp.take(weights, order)
    cum = xp.concat([xp.zeros(1, dtype=w.dtype), xp.cumulative_sum(w)])
    return values, cum


def partial_diameter(values, alpha, weights=None):
    """``PartDiam(nu, 1 - alpha)`` of the (weighted) empirical measure.

    Tohoku Definition 4.1, for the push-forward measure ``nu`` described by
    the sample ``values`` (shape ``(n,)``) with optional point ``weights``:
    the smallest diameter of a Borel set of ``nu``-measure at least
    ``1 - alpha``.  Computed exactly via contiguous windows of the sorted
    values (TMLR Lemma 3.1 and its weighted analogue).
    """
    xp = get_namespace(values, weights)
    values = as_float_array(values, xp)
    if values.ndim != 1:
        raise ValueError("values must be 1-D")
    n = values.shape[0]
    if n == 0:
        return 0.0
    if weights is None:
        c = math.ceil(n * (1.0 - alpha) - _MASS_EPS)  # c_alpha, TMLR Sec. 3.1
        if c <= 1:
            return 0.0
        if c > n:
            return math.inf  # measure cannot reach 1 - alpha < ... > 1
        sorted_values = xp.sort(values)
        return float(xp.min(sorted_values[c - 1 :] - sorted_values[: n - c + 1]))
    w = _validate_weights(weights, n, xp)
    if 1.0 - alpha > 1.0 + _MASS_EPS:
        return math.inf
    sorted_values, cum = _sort_with_weights(values, w, xp)
    out = _weighted_partdiams_at_masses(
        sorted_values, cum, xp.asarray([1.0 - alpha]), xp
    )
    return float(out[0])


def observable_diameter(phi, alpha, weights=None, *, normalize=None):
    """``ObsDiam(D; -alpha) = sup_f PartDiam(f_* mu, 1 - alpha)``.

    Tohoku Definition 4.1 for the finite geometric data set represented by
    the value matrix ``phi`` (shape ``(n_features, n_points)``) and the
    (weighted) empirical measure.  ``normalize`` optionally rescales the
    features by ``tau`` as in Tohoku Section 6.1.1 (see
    :func:`compute_dimension`); the raw definition uses ``normalize=None``.
    """
    xp = get_namespace(phi, weights)
    phi = _validate_phi(phi, xp)
    n_features, n = phi.shape
    if n == 0 or n_features == 0:
        return 0.0
    scale = _resolve_scale(phi, normalize, xp)
    if weights is None:
        c = math.ceil(n * (1.0 - alpha) - _MASS_EPS)
        if c <= 1:
            return 0.0
        if c > n:
            return math.inf
        sorted_phi = xp.sort(phi, axis=1)
        return scale * float(xp.max(_phi_window(sorted_phi, c, xp)))
    w = _validate_weights(weights, n, xp)
    best = 0.0
    for j in range(n_features):
        values, cum = _sort_with_weights(phi[j, :], w, xp)
        out = _weighted_partdiams_at_masses(values, cum, xp.asarray([1.0 - alpha]), xp)
        best = max(best, float(out[0]))
    return scale * best


# ---------------------------------------------------------------------------
# Discriminability Delta and the dimension
# ---------------------------------------------------------------------------


def _uniform_delta_exact(sorted_phi, scale, cap, xp):
    """TMLR Theorem 3.2 / Algorithm 1: exact ``Delta`` for uniform weights."""
    n = sorted_phi.shape[1]
    total = 0.0
    for k in range(2, n + 1):
        obs = scale * float(xp.max(_phi_window(sorted_phi, k, xp)))
        total += min(obs, cap)
    return total / n


def _uniform_delta_support(sorted_phi, s, exact, scale, cap, xp):
    """TMLR Algorithm 2: support-sequence bounds, optional exact refinement.

    Returns ``(delta_minus, delta_plus, delta_exact_or_None)`` following TMLR
    Definition 4.2 and, for the refinement, Lemma 4.7: within a gap
    ``s_i < j < s_{i+1}`` only the features with ``phi_{s_{i+1}, f} >
    phi_{s_i}(D)`` can change the maximum, so all others are skipped.
    """
    n = sorted_phi.shape[1]
    phi_vec = {k: _phi_window(sorted_phi, k, xp) for k in s}  # per-feature
    obs = {k: min(scale * float(xp.max(v)), cap) for k, v in phi_vec.items()}

    delta_minus = sum(obs[k] for k in s)
    delta_plus = delta_minus
    for i in range(len(s) - 1):
        gap = s[i + 1] - s[i] - 1
        delta_minus += gap * obs[s[i]]
        delta_plus += gap * obs[s[i + 1]]
    delta_minus /= n
    delta_plus /= n

    delta_exact = None
    if exact:
        total = sum(obs[k] for k in s)
        for i in range(len(s) - 1):
            lo, hi = s[i], s[i + 1]
            if hi - lo <= 1:
                continue
            # unscaled comparison; scale > 0 preserves the order
            phi_lo_max = float(xp.max(phi_vec[lo]))
            mask = phi_vec[hi] > phi_lo_max
            sub = sorted_phi[mask, :] if bool(xp.any(mask)) else None
            for k in range(lo + 1, hi):
                phi_k = phi_lo_max
                if sub is not None:
                    phi_k = max(phi_k, float(xp.max(_phi_window(sub, k, xp))))
                total += min(scale * phi_k, cap)
        delta_exact = total / n
    return delta_minus, delta_plus, delta_exact


def _weighted_delta_grid(phi, weights, grid, scale, cap, xp):
    """``Delta`` for non-uniform weights via a grid of mass levels.

    ``ObsDiam(D; -alpha)`` as a function of the mass level ``t = 1 - alpha``
    is a non-decreasing step function, so evaluating it on the grid ``t_j =
    j/G`` yields certified bounds: left endpoints give a lower and right
    endpoints an upper Riemann sum for ``Delta = int_0^1 ObsDiam ^ 1 dt``.
    (For the uniform measure the step positions are exactly ``k/n`` and the
    grid ``G = n`` recovers TMLR Theorem 3.2; that case is dispatched to the
    exact integer-window path instead.)
    """
    n_features, n = phi.shape
    masses = xp.linspace(1.0 / grid, 1.0, grid)
    obs = xp.zeros(grid, dtype=phi.dtype)
    for j in range(n_features):
        values, cum = _sort_with_weights(phi[j, :], weights, xp)
        part = _weighted_partdiams_at_masses(values, cum, masses, xp)
        obs = xp.where(part > obs, part, obs)
    obs_np = np.asarray([min(scale * float(o), cap) for o in obs])
    delta_upper = float(np.sum(obs_np)) / grid
    delta_lower = float(np.sum(obs_np[:-1])) / grid  # left endpoints, O(0) = 0
    return delta_lower, delta_upper


def _dimension_from_delta(delta):
    return 1.0 / (delta * delta) if delta > 0 else math.inf


def compute_dimension(
    phi,
    weights=None,
    *,
    normalize=None,
    support_sequence=None,
    exact=False,
    grid=None,
    cap=1.0,
):
    """Compute the intrinsic dimension from feature values alone.

    Implements :math:`\\partial(\\mathcal{D}) = 1/\\Delta(\\mathcal{D})^2`
    (Tohoku Proposition 5.3 / TMLR Eq. (1)) with
    :math:`\\Delta(\\mathcal{D}) = \\int_0^1 \\mathrm{ObsDiam}(\\mathcal{D};
    -\\alpha) \\wedge 1\\, d\\alpha` (Tohoku Proposition 4.5) for the finite
    geometric data set given by ``phi`` and ``weights``.

    Parameters
    ----------
    phi : array of shape (n_features, n_points)
        Value matrix, ``phi[j, i] = f_j(x_i)``.  The functional needs
        nothing else about the features; for the dimension to be *the*
        :math:`\\partial_F` of the papers the features must be 1-Lipschitz
        with respect to the data metric (see :mod:`obsdim` feature
        families).
    weights : array of shape (n_points,), optional
        Point weights, i.e. a non-uniform measure :math:`\\mu` on the
        mm-space.  ``None`` (default) is the normalized counting measure,
        for which the computation is exact (TMLR Theorem 3.2, Lemma 3.5).
        Non-uniform weights are handled by exact partial diameters on a grid
        of ``grid`` mass levels, yielding certified ``delta_bounds``.
    normalize : {"diameter", None} or float, default None
        ``None`` computes the paper functional on the raw features.
        ``"diameter"`` rescales features by :math:`\\tau = 1/\\mathrm{diam}
        (X, d_F)` (Tohoku Section 6.1.1) so that dimensions of data sets
        with different absolute size are comparable; this is what the
        experiments in both papers do.  A positive float is used as the
        diameter to divide by.
    support_sequence : sequence of int or int, optional
        Uniform weights only.  A support sequence ``2 = s_1 < ... < s_l =
        n_points`` (TMLR Definition 4.2); an integer requests that many
        geometrically spaced support points.  Reduces the cost from
        ``O(n_features * n_points^2)`` to ``O(n_features * n_points * l)``
        and returns certified dimension bounds (TMLR Corollary 4.3).
    exact : bool, default False
        With a support sequence: additionally refine to the exact value
        while skipping features that provably cannot contribute (TMLR
        Lemma 4.7 / Algorithm 2).
    grid : int, optional
        Number of mass levels for the non-uniform-weights path (default:
        ``n_points``).  Ignored for uniform weights.
    cap : float or None, default 1.0
        The cap in :math:`\\mathrm{ObsDiam} \\wedge 1` of Tohoku
        Proposition 4.5.  ``None`` drops it (TMLR Eq. (1) convention; the
        two agree whenever features have range at most 1, in particular
        after normalization).

    Returns
    -------
    DimensionResult
    """
    xp = get_namespace(phi, weights)
    phi = _validate_phi(phi, xp)
    n_features, n = phi.shape
    cap = _cap_value(cap)
    scale = _resolve_scale(phi, normalize, xp)

    if weights is not None:
        w = _validate_weights(weights, n, xp)
        if _is_uniform(w, xp):
            weights = None
    if weights is not None and support_sequence is not None:
        raise ValueError(
            "support_sequence requires the uniform measure (weights=None); "
            "use `grid` to control the accuracy for weighted data"
        )

    if n < 2 or n_features == 0:
        delta_lo = delta_hi = delta = 0.0
        is_exact = True
    elif weights is None:
        sorted_phi = xp.sort(phi, axis=1)
        if support_sequence is None:
            delta = _uniform_delta_exact(sorted_phi, scale, cap, xp)
            delta_lo = delta_hi = delta
            is_exact = True
        else:
            s = _make_support_sequence(support_sequence, n)
            delta_lo, delta_hi, delta_exact = _uniform_delta_support(
                sorted_phi, s, exact, scale, cap, xp
            )
            is_exact = delta_exact is not None
            delta = delta_exact if is_exact else 0.5 * (delta_lo + delta_hi)
    else:
        grid = int(grid) if grid is not None else max(n, 2)
        if grid < 1:
            raise ValueError("grid must be a positive integer")
        delta_lo, delta_hi = _weighted_delta_grid(phi, w, grid, scale, cap, xp)
        delta = 0.5 * (delta_lo + delta_hi)
        is_exact = False

    dim = _dimension_from_delta(delta)
    dim_lo = _dimension_from_delta(delta_hi)  # note the inversion
    dim_hi = _dimension_from_delta(delta_lo)
    if is_exact:
        delta_lo = delta_hi = delta
        dim_lo = dim_hi = dim
        error = 0.0
    elif math.isinf(dim_hi):
        error = 0.0 if math.isinf(dim_lo) else math.inf
    else:
        error = (dim_hi - dim_lo) / dim_lo  # E(s, D), TMLR Definition 4.4

    return DimensionResult(
        dimension=dim,
        dimension_bounds=(dim_lo, dim_hi),
        delta=delta,
        delta_bounds=(delta_lo, delta_hi),
        approximation_error=error,
        n_points=n,
        n_features=n_features,
        scale=scale,
        exact=is_exact,
    )


def discriminability(phi, weights=None, **kwargs):
    """The discriminability :math:`\\Delta(\\mathcal{D}) \\in [0, 1]`.

    Tohoku Proposition 4.5 / TMLR Eq. (1) and Theorem 3.2.  Accepts the same
    keyword arguments as :func:`compute_dimension`.
    """
    return compute_dimension(phi, weights, **kwargs).delta


def observable_dimension(phi, weights=None, **kwargs):
    """The intrinsic dimension :math:`\\partial_F` from feature values alone.

    ``phi`` has shape ``(n_features, n_points)`` with ``phi[j, i] =
    f_j(x_i)``; optional ``weights`` describe a non-uniform measure.  Returns
    :math:`\\partial(\\mathcal{D}) = 1/\\Delta(\\mathcal{D})^2` as a float
    (Tohoku Proposition 5.3; TMLR Eq. (1)).  See :func:`compute_dimension`
    for options and for the full result object with certified bounds.
    """
    return compute_dimension(phi, weights, **kwargs).dimension
