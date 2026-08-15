"""Design sketch: intrinsic dimension with pluggable feature families.

Core idea (Hanika/Schneider/Stumme, Tohoku Math. J. 74, 2022): the intrinsic
dimension is parametrized by a family F of (1-Lipschitz) feature functions.
The dimension functional only ever consumes the *values* of those functions
on the sample, so feature families and the dimension computation are fully
decoupled.
"""

from __future__ import annotations

import abc
from typing import Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator


# --------------------------------------------------------------------------
# Layer 1: feature families -> value matrix Phi of shape (k, n)
# --------------------------------------------------------------------------

class FeatureFamily(abc.ABC):
    """A family F of real-valued feature functions on the data.

    Contract: `evaluate` returns Phi with Phi[j, i] = f_j(x_i). If the family
    is infinite/parametrized, `evaluate` may return a Monte-Carlo sample of
    `n_functions` members (the large-scale TMLR-2023 setting).
    """

    #: Declared Lipschitz bound of every member w.r.t. the data metric.
    #: Built-in families guarantee this by construction.
    lipschitz: float = 1.0

    @abc.abstractmethod
    def evaluate(self, X, *, n_functions=None, rng=None) -> np.ndarray:
        ...


class OnePointDistances(FeatureFamily):
    """F = { d(x, .) : x anchor }  -- the canonical family of the paper.

    metric:  "euclidean", a callable d(X, Y), or "precomputed" (X is the
             distance matrix itself), which covers graphs / general mm-spaces.
    anchors: "all" reproduces the exact Tohoku computation (O(n^2));
             an int m samples m anchors -> the scalable approximation.
    """

    def __init__(self, metric="euclidean", anchors="all"):
        self.metric = metric
        self.anchors = anchors

    def evaluate(self, X, *, n_functions=None, rng=None):
        rng = np.random.default_rng(rng)
        n = X.shape[0]
        m = self.anchors if isinstance(self.anchors, int) else n
        idx = np.arange(n) if m == n else rng.choice(n, size=m, replace=False)
        if self.metric == "precomputed":
            return X[idx]                      # rows of the distance matrix
        if callable(self.metric):
            return self.metric(X[idx], X)
        # euclidean default; swap in scipy.spatial.distance.cdist in practice
        return np.linalg.norm(X[idx, None, :] - X[None, :, :], axis=-1)


class CustomFeatures(FeatureFamily):
    """User-supplied functions: vectorized callables f(X) -> shape (n,).

    lipschitz="estimate" rescales by an empirical Lipschitz constant computed
    from sampled pairs; lipschitz=None disables normalization (dimension then
    only comparable within this fixed family); a float declares the bound.
    """

    def __init__(self, funcs: Sequence[Callable], lipschitz="estimate",
                 metric="euclidean"):
        self.funcs = list(funcs)
        self.lipschitz_mode = lipschitz
        self.metric = metric

    def evaluate(self, X, *, n_functions=None, rng=None):
        phi = np.stack([np.asarray(f(X), dtype=float) for f in self.funcs])
        if self.lipschitz_mode == "estimate":
            phi = phi / self._empirical_lipschitz(phi, X, rng)[:, None]
        elif isinstance(self.lipschitz_mode, (int, float)):
            phi = phi / float(self.lipschitz_mode)
        return phi

    def _empirical_lipschitz(self, phi, X, rng, n_pairs=10_000):
        rng = np.random.default_rng(rng)
        i = rng.integers(0, X.shape[0], n_pairs)
        j = rng.integers(0, X.shape[0], n_pairs)
        d = np.linalg.norm(X[i] - X[j], axis=-1)  # respect self.metric in real impl
        keep = d > 0
        ratios = np.abs(phi[:, i[keep]] - phi[:, j[keep]]) / d[keep]
        return np.maximum(ratios.max(axis=1), np.finfo(float).tiny)


class PrecomputedFeatures(FeatureFamily):
    """The identity plugin: user hands over Phi directly.

    Strongest expression of the paper's generality -- features need not be
    functions of any coordinates available to the package (kernel columns,
    model logits, graph centralities, ...).
    """

    def __init__(self, phi, lipschitz=1.0):
        self.phi = np.asarray(phi, dtype=float)
        self.lipschitz = lipschitz

    def evaluate(self, X=None, *, n_functions=None, rng=None):
        return self.phi


class UnionFamily(FeatureFamily):
    """F = F_1 ∪ ... ∪ F_m, e.g. distances plus a handful of domain features."""

    def __init__(self, families: Sequence[FeatureFamily]):
        self.families = list(families)

    def evaluate(self, X, *, n_functions=None, rng=None):
        return np.vstack([f.evaluate(X, n_functions=n_functions, rng=rng)
                          for f in self.families])


# --------------------------------------------------------------------------
# Layer 2: dimension functional -- consumes only (Phi, weights)
# --------------------------------------------------------------------------

def observable_dimension(phi: np.ndarray, weights=None) -> float:
    """Compute the F-intrinsic dimension from feature values alone.

    Pipeline per the papers: for each kappa in a grid, take the sup over
    features of the (1-kappa)-partial diameter of the push-forward of the
    (weighted) empirical measure, then integrate the squared observable
    diameters over kappa and invert/normalize.  (Stub -- fill in the exact
    functional from Tohoku Def./TMLR Sec. 3.)
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# sklearn-facing estimator
# --------------------------------------------------------------------------

class GeometricID(BaseEstimator):
    """Intrinsic dimension estimator parametrized by a feature family.

    >>> GeometricID().fit(X).dimension_                    # one-point dists
    >>> GeometricID(OnePointDistances(anchors=64)).fit(X)  # scalable approx.
    >>> GeometricID(CustomFeatures([f, g, h])).fit(X)      # custom family
    >>> GeometricID(PrecomputedFeatures(Phi)).fit(None)    # values only
    """

    def __init__(self, features: FeatureFamily | None = None,
                 sample_weight_mode=None, random_state=None):
        self.features = features
        self.sample_weight_mode = sample_weight_mode
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        family = self.features or OnePointDistances()
        phi = family.evaluate(X, rng=self.random_state)
        self.dimension_ = observable_dimension(phi, sample_weight)
        return self
