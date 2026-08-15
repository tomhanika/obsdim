"""Quickstart: intrinsic dimension of spheres.

The intrinsic dimension of the papers grows linearly with the "true"
dimension of the underlying space (axiom of geometric order of divergence,
Tohoku Cor. 5.5): higher-dimensional spheres concentrate more, discriminate
less, and get a higher ID.
"""

import numpy as np

from obsdim import GeometricID

rng = np.random.default_rng(0)

for d in (2, 4, 8, 16, 32):
    x = rng.normal(size=(400, d + 1))
    x /= np.linalg.norm(x, axis=1, keepdims=True)  # uniform sample of S^d
    est = GeometricID().fit(x)  # one-point distances, exact O(n^2)
    print(
        f"S^{d:<2}  dimension = {est.dimension_:6.2f}   "
        f"discriminability Delta = {est.discriminability_:.3f}"
    )

# For large data, sample anchors and use a support sequence (TMLR 2023):
# certified bounds instead of a point value, at O(m * n * l) cost.
from obsdim import OnePointDistances  # noqa: E402

x = rng.normal(size=(5000, 16))
est = GeometricID(
    OnePointDistances(anchors=256),  # m = 256 sampled distance functions
    support_sequence=64,  # l = 64 support points (TMLR Def. 4.2)
    random_state=0,
).fit(x)
lo, hi = est.dimension_bounds_
print(
    f"\nGaussian in R^16, n=5000:  {lo:.1f} <= dimension <= {hi:.1f} "
    f"(error bound {est.approximation_error_:.1%})"
)
