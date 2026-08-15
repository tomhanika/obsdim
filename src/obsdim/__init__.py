"""obsdim: concentration-based intrinsic dimension of geometric data sets.

Reference implementation of the intrinsic dimension :math:`\\partial_F` of

- Hanika, Schneider, Stumme, *Intrinsic dimension of geometric data sets*,
  Tohoku Math. J. 74 (2022) 23-52, doi:10.2748/tmj.20201015a, and
- Stubbemann, Hanika, Schneider, *Intrinsic Dimension for Large-Scale
  Geometric Learning*, TMLR 2023 (scalable algorithms).

The dimension is parametrized by an exchangeable family ``F`` of
(1-Lipschitz) feature functions; see the feature-family classes.  Typical
use::

    from obsdim import GeometricID, OnePointDistances

    GeometricID().fit(X).dimension_                       # exact, O(n^2)
    GeometricID(OnePointDistances(anchors=256),
                support_sequence=32).fit(X).dimension_    # large-scale
"""

from ._core import (
    DimensionResult,
    compute_dimension,
    discriminability,
    observable_diameter,
    observable_dimension,
    partial_diameter,
)
from ._estimator import GeometricID
from ._families import (
    CoordinateProjections,
    CustomFeatures,
    FeatureFamily,
    FeatureScaler,
    OnePointDistances,
    PrecomputedFeatures,
    UnionFamily,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "CoordinateProjections",
    "CustomFeatures",
    "DimensionResult",
    "FeatureFamily",
    "FeatureScaler",
    "GeometricID",
    "OnePointDistances",
    "PrecomputedFeatures",
    "UnionFamily",
    "compute_dimension",
    "discriminability",
    "observable_diameter",
    "observable_dimension",
    "partial_diameter",
]
