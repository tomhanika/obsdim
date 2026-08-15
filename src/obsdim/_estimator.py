"""scikit-learn estimator wrapping the dimension functional."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from ._core import compute_dimension
from ._families import FeatureFamily, OnePointDistances

__all__ = ["GeometricID"]


class GeometricID(BaseEstimator):
    """Concentration-based intrinsic dimension, parametrized by a feature family.

    Implements :math:`\\partial_F(\\mathcal{D}) = 1/\\Delta(\\mathcal{D})^2`
    of Hanika, Schneider, Stumme (Tohoku Math. J. 74, 2022, Prop. 5.3) with
    the finite-data algorithms of Stubbemann, Hanika, Schneider (TMLR 2023).
    The estimator follows the ``fit(X) -> .dimension_`` convention of
    ``scikit-dimension``, so it drops into existing ID benchmark scripts.

    Parameters
    ----------
    features : FeatureFamily, optional
        The feature family ``F`` parametrizing the dimension.  Default:
        ``OnePointDistances()`` — Euclidean distance functions to all data
        points, the canonical (and exact, ``O(n^2)``) choice of the papers.
    normalize : {"diameter", None} or float, default "diameter"
        Feature rescaling by ``tau = 1/diameter`` (Tohoku Section 6.1.1),
        making dimensions comparable across data sets of different absolute
        size; this matches the experimental methodology of both papers.
        ``None`` evaluates the raw functional of the definitions instead.
    support_sequence : sequence of int or int, optional
        Support sequence for the scalable approximation (TMLR Def. 4.2); an
        int requests that many geometrically spaced support points.  Yields
        certified ``dimension_bounds_`` at ``O(n_features * n * l)`` cost.
    exact : bool, default False
        With a support sequence: refine to the exact dimension via TMLR
        Lemma 4.7 / Algorithm 2.
    grid : int, optional
        Accuracy (number of mass levels) when ``sample_weight`` is
        non-uniform; default ``n``.
    random_state : int, Generator, or None
        Controls anchor sampling / Lipschitz-constant estimation in the
        feature family, if any.

    Attributes
    ----------
    dimension_ : float
        The intrinsic dimension :math:`\\partial_F` (``inf`` for data sets
        that no feature can discriminate, e.g. a single point).
    discriminability_ : float
        :math:`\\Delta(\\mathcal{D}) \\in [0, 1]`.
    dimension_bounds_ : tuple of float
        Certified interval containing the exact dimension (equal endpoints
        for exact computations).
    approximation_error_ : float
        Relative error bound ``E(s, D)`` (TMLR Def. 4.4); ``0.0`` if exact.
    result_ : DimensionResult
        The full result object.

    Examples
    --------
    >>> import numpy as np
    >>> from obsdim import GeometricID, OnePointDistances
    >>> X = np.random.default_rng(0).normal(size=(200, 8))
    >>> GeometricID().fit(X).dimension_                    # doctest: +SKIP
    >>> GeometricID(OnePointDistances(anchors=64)).fit(X)  # doctest: +SKIP
    """

    def __init__(
        self,
        features: FeatureFamily | None = None,
        *,
        normalize="diameter",
        support_sequence=None,
        exact: bool = False,
        grid: int | None = None,
        random_state=None,
    ):
        self.features = features
        self.normalize = normalize
        self.support_sequence = support_sequence
        self.exact = exact
        self.grid = grid
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        """Compute the intrinsic dimension of ``X``.

        ``X`` is ``(n_samples, n_features)`` data, a precomputed distance
        matrix (with ``OnePointDistances(metric="precomputed")``), or
        ``None`` for families that need no data
        (:class:`~obsdim.PrecomputedFeatures`).  ``sample_weight`` realizes
        a non-uniform measure on the mm-space.
        """
        family = self.features if self.features is not None else OnePointDistances()
        needs_data = getattr(family, "requires_data", True)

        if needs_data:
            if isinstance(X, np.ndarray) or not hasattr(X, "__array_namespace__"):
                X = validate_data(
                    self,
                    X,
                    dtype=[np.float64, np.float32],
                    ensure_min_samples=1,
                )
        if sample_weight is not None:
            n = X.shape[0] if needs_data else np.asarray(family.phi).shape[1]
            sample_weight = _check_sample_weight(
                sample_weight, np.empty((n, 1)), dtype=np.float64
            )

        phi = family.evaluate(X, rng=self.random_state)
        result = compute_dimension(
            phi,
            sample_weight,
            normalize=self.normalize,
            support_sequence=self.support_sequence,
            exact=self.exact,
            grid=self.grid,
        )
        self.result_ = result
        self.dimension_ = result.dimension
        self.discriminability_ = result.delta
        self.dimension_bounds_ = result.dimension_bounds
        self.approximation_error_ = result.approximation_error
        return self

    def fit_predict_dimension(self, X, y=None, sample_weight=None):
        """Convenience: ``fit(X)`` and return ``dimension_``."""
        return self.fit(X, y, sample_weight=sample_weight).dimension_

    def __sklearn_is_fitted__(self):
        return hasattr(self, "dimension_")

    def score(self, X=None, y=None):
        """Negative dimension, so that model selection prefers lower ID."""
        check_is_fitted(self)
        if X is not None and hasattr(self, "n_features_in_"):
            validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        return -self.dimension_
