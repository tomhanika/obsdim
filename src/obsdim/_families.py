"""Feature families: pluggable generators of the value matrix ``Phi``.

The Tohoku paper defines the intrinsic dimension of a *geometric data set*
:math:`\\mathcal{D} = (X, F, \\mu)` — the dimension :math:`\\partial_F` is
parametrized by the feature set :math:`F`, and exchanging :math:`F` is a
core idea of the theory (Tohoku Definition 3.1).  The dimension functional
in :mod:`obsdim._core` only ever consumes the value matrix
``Phi[j, i] = f_j(x_i)``; the classes here produce that matrix.

Two equivalent readings of "1-Lipschitz" (both from Tohoku Def. 3.1):

- If a metric ``d`` on the data is *given* (e.g. Euclidean, or a graph
  metric), the axiomatics ask for :math:`F \\subseteq \\mathrm{Lip}_1(X, d)`.
  ``OnePointDistances`` guarantees this by construction (triangle
  inequality); ``CustomFeatures`` offers estimation or declaration of a
  Lipschitz constant to rescale user functions.
- If no metric is given, the features *induce* one:
  :math:`d_F(x, y) = \\sup_{f \\in F} |f(x) - f(y)|`, and every feature is
  automatically 1-Lipschitz with respect to it.  ``PrecomputedFeatures``
  computes :math:`\\partial_F` of that induced geometric data set.

Only in the first reading with verified constants is the result *the*
:math:`\\partial_F` of the paper for ``(X, d)``; otherwise it is the
dimension of the user-defined family, comparable within that family.

**Choosing a normalization.**  Rescaling all features by one global constant
(``normalize="diameter"`` in the functional; Tohoku Section 6.1.1) is a mere
change of units — the underlying mm-space is untouched.  Rescaling *per
feature* or *per sub-family* changes the induced metric ``d_F``, i.e. the
geometric data set itself, and therefore lives here in the family layer,
where the provenance of the features is known.  Rule of thumb:

- features sharing a codomain scale (any metric input, distance functions):
  keep them raw and normalize globally in the functional;
- tabular columns with incommensurable units: per-column scaling at family
  construction (:class:`CoordinateProjections`, the default there), which
  computes :math:`\\partial_F` for the sup-metric on min-max-scaled columns;
- unions of heterogeneous sources: per-member scaling
  (``UnionFamily(..., normalize_members=True)``), which keeps each family's
  internal relative scales but makes the members commensurable.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence

import numpy as np

from ._compat import as_float_array, get_namespace

__all__ = [
    "CoordinateProjections",
    "CustomFeatures",
    "FeatureFamily",
    "FeatureScaler",
    "OnePointDistances",
    "PrecomputedFeatures",
    "UnionFamily",
]


def _rescale_rows(phi, xp, *, per_feature):
    """Divide rows of ``phi`` by their value range (or all by the largest).

    Constant features (range 0) are left untouched: they are neutral for the
    dimension anyway (TMLR Lemma 3.3), and dividing by 0 is undefined.
    """
    ranges = xp.max(phi, axis=1) - xp.min(phi, axis=1)
    one = xp.asarray(1.0, dtype=phi.dtype)
    if per_feature:
        divisors = xp.where(ranges > 0, ranges, one)
        return phi / divisors[:, None]
    diam = xp.max(ranges) if phi.shape[0] else one
    return phi / diam if float(diam) > 0 else phi


def _as_rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        # accept sklearn-style RandomState for interoperability
        return np.random.default_rng(rng.randint(np.iinfo(np.int32).max))
    return np.random.default_rng(rng)


class FeatureFamily(abc.ABC):
    """A family ``F`` of real-valued feature functions on the data.

    Contract: :meth:`evaluate` returns ``Phi`` with ``Phi[j, i] =
    f_j(x_i)``.  If the family is infinite or parametrized, ``evaluate`` may
    return a Monte-Carlo sample of ``n_functions`` members (the large-scale
    setting; note that sub-sampling features can only decrease
    :math:`\\Delta` and hence over-estimate the dimension of the full
    family, by the axiom of feature antitonicity, Tohoku Definition 5.1).
    """

    #: Declared Lipschitz bound of every member w.r.t. the data metric.
    #: Built-in families guarantee this by construction.
    lipschitz: float = 1.0

    #: Whether ``evaluate`` needs the data ``X`` (False for precomputed Phi).
    requires_data: bool = True

    @abc.abstractmethod
    def evaluate(self, X, *, n_functions=None, rng=None):
        """Return the value matrix ``Phi`` of shape ``(k, n_points)``."""


class OnePointDistances(FeatureFamily):
    """``F = { d(x, .) : x anchor }`` — the canonical family of the papers.

    This is the family :math:`\\mathcal{X}_\\circ` of Tohoku Definition 3.2
    and the one used for all experiments there; distance functions are
    1-Lipschitz by the triangle inequality, so the result is *the*
    :math:`\\partial_F` for the chosen metric.

    Parameters
    ----------
    metric : str, callable, or "precomputed", default "euclidean"
        ``"precomputed"`` means ``X`` passed to :meth:`evaluate` *is* the
        distance matrix (rows = anchors) — this covers graphs and general
        mm-spaces and needs no coordinates.  A callable must map
        ``(anchors, X) -> distances`` of shape ``(m, n)``.  Other strings
        are forwarded to :func:`scipy.spatial.distance.cdist` (NumPy inputs;
        ``"euclidean"`` also runs natively on any Array API input).
    anchors : "all" or int, default "all"
        ``"all"`` uses every data point as an anchor — the exact computation
        of the Tohoku paper, ``O(n^2)`` distances.  An integer ``m`` samples
        ``m`` anchors uniformly without replacement — the scalable
        Monte-Carlo variant (see :class:`FeatureFamily` on the induced
        bias direction).
    """

    def __init__(self, metric="euclidean", anchors="all"):
        self.metric = metric
        self.anchors = anchors

    def _anchor_indices(self, n, rng):
        if self.anchors == "all":
            return None
        m = int(self.anchors)
        if not 1 <= m <= n:
            raise ValueError(f"anchors must be in [1, {n}], got {m}")
        if m == n:
            return None
        return np.sort(_as_rng(rng).choice(n, size=m, replace=False))

    def evaluate(self, X, *, n_functions=None, rng=None):
        xp = get_namespace(X)
        X = as_float_array(X, xp)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_points, n_dims) or a distance matrix")
        n = X.shape[0]
        idx = self._anchor_indices(n, rng)
        if self.metric == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    "metric='precomputed' expects a square distance matrix, "
                    f"got shape {X.shape}"
                )
            return X if idx is None else X[xp.asarray(idx), :]
        anchors = X if idx is None else X[xp.asarray(idx), :]
        if callable(self.metric):
            return as_float_array(self.metric(anchors, X), xp)
        if self.metric == "euclidean":
            diff = anchors[:, None, :] - X[None, :, :]
            return xp.sqrt(xp.sum(diff * diff, axis=-1))
        from scipy.spatial.distance import cdist

        return xp.asarray(cdist(np.asarray(anchors), np.asarray(X), self.metric))


class CustomFeatures(FeatureFamily):
    """User-supplied feature functions: vectorized callables ``f(X) -> (n,)``.

    Lipschitz normalization (the axiomatics require 1-Lipschitz features):

    - ``lipschitz="estimate"``: rescale each feature by an empirical
      Lipschitz constant ``max |f(x_i) - f(x_j)| / d(x_i, x_j)`` over
      ``n_pairs`` sampled pairs, measured in the family's ``metric``.  This
      is a lower bound on the true constant, so the result can still
      slightly over-discriminate; it makes dimensions approximately
      comparable across families.
    - a float: every feature is divided by this declared bound;
    - a sequence of floats: per-feature declared bounds;
    - ``None``: no rescaling — the dimension is then that of the raw family,
      only comparable within it (see the module docstring).
    """

    def __init__(
        self,
        funcs: Sequence[Callable],
        lipschitz="estimate",
        metric="euclidean",
        n_pairs: int = 10_000,
    ):
        self.funcs = list(funcs)
        self.lipschitz_mode = lipschitz
        self.metric = metric
        self.n_pairs = n_pairs

    def _pair_distances(self, X, i, j, xp):
        if self.metric == "precomputed":
            return X[xp.asarray(i), xp.asarray(j)]
        if callable(self.metric):
            d = self.metric(X[xp.asarray(i), :], X[xp.asarray(j), :])
            return as_float_array(d, xp)
        if self.metric == "euclidean":
            diff = X[xp.asarray(i), :] - X[xp.asarray(j), :]
            return xp.sqrt(xp.sum(diff * diff, axis=-1))
        from scipy.spatial.distance import cdist

        a, b = np.asarray(X)[i], np.asarray(X)[j]
        return xp.asarray(
            np.array(
                [
                    cdist(a[k : k + 1], b[k : k + 1], self.metric)[0, 0]
                    for k in range(len(i))
                ]
            )
        )

    def _empirical_lipschitz(self, phi, X, rng, xp):
        n = phi.shape[1]
        rng = _as_rng(rng)
        n_pairs = min(self.n_pairs, 4 * n * n)
        i = rng.integers(0, n, n_pairs)
        j = rng.integers(0, n, n_pairs)
        d = self._pair_distances(X, i, j, xp)
        keep = d > 0
        if not bool(xp.any(keep)):
            return xp.ones(phi.shape[0], dtype=phi.dtype)
        num = xp.abs(phi[:, xp.asarray(i)[keep]] - phi[:, xp.asarray(j)[keep]])
        ratios = num / d[keep]
        constants = xp.max(ratios, axis=1)
        tiny = xp.asarray(np.finfo(np.float64).tiny, dtype=phi.dtype)
        return xp.where(constants > tiny, constants, xp.ones_like(constants))

    def evaluate(self, X, *, n_functions=None, rng=None):
        xp = get_namespace(X)
        rows = [as_float_array(f(X), xp) for f in self.funcs]
        phi = xp.stack(rows)
        if phi.ndim != 2:
            raise ValueError("each feature must return a 1-D array of values")
        mode = self.lipschitz_mode
        if mode is None:
            return phi
        if mode == "estimate":
            constants = self._empirical_lipschitz(phi, X, rng, xp)
        elif np.isscalar(mode):
            constants = xp.full(phi.shape[0], float(mode), dtype=phi.dtype)
        else:
            constants = as_float_array(mode, xp)
            if constants.shape != (phi.shape[0],):
                raise ValueError(
                    "per-feature lipschitz bounds must match the number of funcs"
                )
        return phi / constants[:, None]


class PrecomputedFeatures(FeatureFamily):
    """The identity plugin: the user hands over ``Phi`` directly.

    Strongest expression of the theory's generality — features need not be
    functions of any coordinates the package sees (kernel columns, model
    logits, graph centralities, ...).  The computed dimension is
    :math:`\\partial_F` of the geometric data set induced by these values
    (metric :math:`d_F`, see the module docstring), or of ``(X, d)`` if the
    caller guarantees ``lipschitz``-Lipschitz continuity w.r.t. their
    metric ``d`` (values are divided by ``lipschitz``).
    """

    requires_data = False

    def __init__(self, phi, lipschitz: float = 1.0):
        self.phi = phi
        self.lipschitz = float(lipschitz)

    def evaluate(self, X=None, *, n_functions=None, rng=None):
        xp = get_namespace(self.phi)
        phi = as_float_array(self.phi, xp)
        if phi.ndim != 2:
            raise ValueError("phi must be 2-D (n_features, n_points)")
        if self.lipschitz != 1.0:
            phi = phi / self.lipschitz
        return phi


class CoordinateProjections(FeatureFamily):
    """``F = { x -> x_j }`` — coordinate projections, the tabular family.

    Parameters
    ----------
    scale : {"range", "global", None}, default "range"
        ``"range"`` (default) divides each column by its own value range
        (min-max scaling).  This is the right choice when columns carry
        incommensurable units: the resulting features are exactly the
        1-Lipschitz coordinate functions of the **sup-metric on
        min-max-scaled columns**, i.e. the induced metric is
        ``d_F(x, y) = max_j |x~_j - y~_j|`` — the computed dimension is
        :math:`\\partial_F` of that geometric data set.  ``"global"``
        divides all columns by the largest column range instead (one common
        unit; the convention of TMLR Definition 5.1 / ID4GeoL for attribute
        vectors that *do* share units) — a pure change of units that keeps
        the columns' relative scales.  ``None`` uses raw values.

    Notes
    -----
    Per-column scaling changes the induced metric (not merely its unit), so
    dimensions computed with ``scale="range"`` are comparable across
    independent affine rescalings of the columns, but are *not* the
    :math:`\\partial_F` of the raw-coordinate sup-metric.  See the module
    docstring for the normalization decision rules.
    """

    def __init__(self, scale="range"):
        self.scale = scale

    def evaluate(self, X, *, n_functions=None, rng=None):
        xp = get_namespace(X)
        X = as_float_array(X, xp)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_points, n_dims)")
        phi = X.T
        if self.scale is None:
            return phi
        if self.scale == "range":
            return _rescale_rows(phi, xp, per_feature=True)
        if self.scale == "global":
            return _rescale_rows(phi, xp, per_feature=False)
        raise ValueError("scale must be 'range', 'global', or None")


class FeatureScaler(FeatureFamily):
    """Wrap any family and rescale the codomains of its features.

    ``per_feature=True`` divides every feature by its own value range,
    making all features count equally in the observable diameter.  This
    **changes the induced metric** ``d_F`` (it is a feature-engineering
    step, not a change of units) — e.g. for distance functions it amplifies
    central anchors relative to peripheral ones; prefer it only when the
    wrapped features have genuinely incommensurable codomains.

    ``per_feature=False`` divides all features by the largest range, i.e. by
    the diameter of ``(X, d_F)`` — the same operation as
    ``normalize="diameter"`` in the dimension functional (Tohoku
    Section 6.1.1), provided here so it can be applied per member inside a
    :class:`UnionFamily`.
    """

    def __init__(self, family: FeatureFamily, per_feature: bool = True):
        self.family = family
        self.per_feature = per_feature

    @property
    def requires_data(self):  # type: ignore[override]
        return self.family.requires_data

    def evaluate(self, X=None, *, n_functions=None, rng=None):
        phi = self.family.evaluate(X, n_functions=n_functions, rng=rng)
        xp = get_namespace(phi)
        return _rescale_rows(phi, xp, per_feature=self.per_feature)


class UnionFamily(FeatureFamily):
    """``F = F_1 ∪ ... ∪ F_m`` — e.g. distances plus domain features.

    Unions can only increase discriminability, hence decrease the dimension
    (axiom of feature antitonicity, Tohoku Definition 5.1(3)).

    Parameters
    ----------
    families : sequence of FeatureFamily
    normalize_members : bool, default False
        Divide each member family's features by that member's diameter
        (its largest feature range) before taking the union.  This keeps
        every family's internal relative scales — its geometric meaning —
        while making heterogeneous members commensurable, so no member
        dominates the observable diameter merely by having a larger
        codomain.  Equivalent to wrapping each member in
        ``FeatureScaler(member, per_feature=False)``.  Note that with
        member-wise scaling the union is a different geometric data set
        than the union of the raw families.
    """

    def __init__(
        self, families: Sequence[FeatureFamily], normalize_members: bool = False
    ):
        self.families = list(families)
        self.normalize_members = normalize_members

    @property
    def requires_data(self):  # type: ignore[override]
        return any(f.requires_data for f in self.families)

    def evaluate(self, X=None, *, n_functions=None, rng=None):
        blocks = [
            f.evaluate(X, n_functions=n_functions, rng=rng) for f in self.families
        ]
        xp = get_namespace(*blocks)
        if self.normalize_members:
            blocks = [_rescale_rows(b, xp, per_feature=False) for b in blocks]
        return xp.concat(blocks, axis=0)
