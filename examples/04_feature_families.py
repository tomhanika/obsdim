"""The feature family is exchangeable — the headline idea of the theory.

The dimension \\partial_F is *parametrized* by the feature set F (Tohoku
Def. 3.1).  The functional only ever sees the value matrix
Phi[j, i] = f_j(x_i), so anything that yields values can be a family:
distance functions, projections, kernel columns, model logits, ...
"""

import numpy as np

from obsdim import (
    CustomFeatures,
    GeometricID,
    OnePointDistances,
    PrecomputedFeatures,
    UnionFamily,
    observable_dimension,
)

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 5))

# 1. Canonical family: one-point distance functions (1-Lipschitz for free).
print("distances:        ", GeometricID().fit(X).dimension_)

# 2. Custom 1-Lipschitz functions.  Declare the constant, or let the package
#    estimate it empirically from sampled pairs and rescale.
funcs = [
    lambda X: X @ np.ones(5) / np.sqrt(5),  # unit-vector projection, L = 1
    lambda X: np.linalg.norm(X, axis=1),  # distance to origin,     L = 1
]
fam = CustomFeatures(funcs, lipschitz="estimate")
print("custom features:  ", GeometricID(fam, random_state=0).fit(X).dimension_)

# 3. Features that are no functions of the coordinates at all: hand over
#    the value matrix.  Here: a random-kernel embedding of the data.
phi = np.tanh(rng.normal(size=(20, 5)) @ X.T)  # (20 features, 300 points)
print("precomputed Phi:  ", GeometricID(PrecomputedFeatures(phi)).fit(None).dimension_)

# 4. Unions: "distances plus domain knowledge".  More features can only
#    lower the dimension (feature antitonicity).  With members of different
#    codomain scales, normalize per member so both actually contribute:
union = UnionFamily(
    [OnePointDistances(), PrecomputedFeatures(phi)],
    normalize_members=True,  # each member scaled by its own diameter
)
dim_union = GeometricID(union).fit(X).dimension_
print("union of both:    ", dim_union)
# The union is at most as high as its lowest member; here the kernel
# features discriminate better at every mass level, so the union's
# observable diameters -- and hence its dimension -- coincide with theirs.

# The low-level route without the estimator: families produce Phi, the
# functional consumes it.
phi_dist = OnePointDistances().evaluate(X)
assert np.isclose(
    observable_dimension(phi_dist, normalize="diameter"),
    GeometricID().fit(X).dimension_,
)
