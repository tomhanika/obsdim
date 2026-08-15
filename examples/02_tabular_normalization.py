"""Normalization for tabular data: which scaling, and why it matters.

Rule of thumb:

- Features sharing one codomain scale (distances in a metric space): keep
  them raw, normalize *globally* — a pure change of units
  (``normalize="diameter"``, Tohoku Sec. 6.1.1).  This is the default of
  ``GeometricID``.
- Table columns with incommensurable units (kg vs EUR): scale *per column*
  at family construction — ``CoordinateProjections(scale="range")``.  This
  computes the dimension for the sup-metric on min-max-scaled columns, and
  is invariant under per-column affine changes of units.
"""

import numpy as np

from obsdim import CoordinateProjections, GeometricID

rng = np.random.default_rng(0)

# A table whose columns carry different units: the second column is the
# same signal, just expressed in "grams" instead of "kilograms".
base = rng.normal(size=(300, 3))
table_kg = base * np.array([1.0, 1.0, 1.0])
table_g = base * np.array([1.0, 1000.0, 1.0])  # column 1 rescaled only

# Global normalization is unit-blind ACROSS columns: the large column
# dominates and the dimension changes with an arbitrary choice of unit.
for name, tab in [("kg", table_kg), ("g ", table_g)]:
    d = GeometricID(CoordinateProjections(scale="global")).fit(tab).dimension_
    print(f"scale='global', column 1 in {name}: dimension = {d:7.2f}")

# Per-column scaling is invariant under per-column units — the honest
# choice when columns are incommensurable.
for name, tab in [("kg", table_kg), ("g ", table_g)]:
    d = GeometricID(CoordinateProjections(scale="range")).fit(tab).dimension_
    print(f"scale='range',  column 1 in {name}: dimension = {d:7.2f}")

# Caveat: per-column scaling is not a change of units but a change of
# geometry -- it computes \partial_F of a different geometric data set
# (sup-metric on min-max-scaled columns).  Only compare dimensions computed
# with the same convention.
