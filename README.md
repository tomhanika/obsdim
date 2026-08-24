# obsdim

[![CI](https://github.com/tomhanika/obsdim/actions/workflows/ci.yml/badge.svg)](https://github.com/tomhanika/obsdim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/obsdim)](https://pypi.org/project/obsdim/)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)

**Concentration-based intrinsic dimension of geometric data sets.**

`obsdim` is the maintained, pip-installable reference implementation of the
intrinsic dimension ∂<sub>F</sub> introduced in

- Tom Hanika, Friedrich Martin Schneider, Gerd Stumme,
  *Intrinsic dimension of geometric data sets*,
  Tohoku Mathematical Journal 74 (2022) 23–52.
  [doi:10.2748/tmj.20201015a](https://doi.org/10.2748/tmj.20201015a)
  ([arXiv:1801.07985](https://arxiv.org/abs/1801.07985))
- Maximilian Stubbemann, Tom Hanika, Friedrich Martin Schneider,
  *Intrinsic Dimension for Large-Scale Geometric Learning*, TMLR 2023.
  ([OpenReview](https://openreview.net/forum?id=85BfDdYMBY),
  [arXiv:2210.05301](https://arxiv.org/abs/2210.05301))

The dimension is defined through the concentration of measure phenomenon
(following V. Pestov): a geometric data set is a triple (X, F, μ) of points,
a family F of 1-Lipschitz *feature functions*, and a probability measure μ.
Features that concentrate — that fail to discriminate points — signal high
intrinsic dimension:

```
∂(𝒟) = 1 / Δ(𝒟)²,     Δ(𝒟) = ∫₀¹ ObsDiam(𝒟; −α) ∧ 1 dα
```

where the observable diameter `ObsDiam` is the largest diameter that any
feature's push-forward of μ retains after discarding an α-fraction of mass.
Every function/class documents the exact definition, theorem, or algorithm
of the papers it implements. Numerical results are cross-validated against
the authors' experiment code, [ID4GeoL](https://github.com/mstubbemann/ID4GeoL).

## Install

```bash
pip install obsdim               # not yet published; for now:
pip install -e .
```

Hard dependencies: `numpy`, `scipy`, `scikit-learn`. Extras:
`obsdim[array-api]` (PyTorch/CuPy/JAX array support via `array-api-compat`).

## Quickstart

```python
import numpy as np
from obsdim import GeometricID

X = np.random.default_rng(0).normal(size=(500, 16))
est = GeometricID().fit(X)      # scikit-learn conventions, skdim-compatible
est.dimension_                  # the intrinsic dimension ∂_F
```

The default is the canonical setting of the papers: one-point distance
functions `d(x, ·)` as features, exact O(n²) computation, features
normalized by the data diameter for comparability across data sets.

### Large-scale data (TMLR 2023 algorithms)

```python
from obsdim import GeometricID, OnePointDistances

est = GeometricID(
    OnePointDistances(anchors=256),   # Monte-Carlo feature sampling
    support_sequence=32,              # TMLR support-sequence approximation
).fit(X)
est.dimension_bounds_        # certified interval for the exact dimension
est.approximation_error_     # relative error bound E(s, 𝒟), TMLR Def. 4.4
# support_sequence=32, exact=True refines to the exact value (TMLR Alg. 2)
```

### The feature family is exchangeable — that's the point

∂<sub>F</sub> is parametrized by F. The dimension functional only ever sees
the value matrix `Phi[j, i] = f_j(x_i)`, so any feature family plugs in:

```python
from obsdim import CustomFeatures, PrecomputedFeatures, UnionFamily

# arbitrary 1-Lipschitz features (constants estimated or declared)
GeometricID(CustomFeatures([f, g, h], lipschitz="estimate")).fit(X)

# features that are not functions of coordinates at all:
# kernel columns, model logits, graph centralities, ...
GeometricID(PrecomputedFeatures(phi), normalize=None).fit(None)

# "distances plus domain knowledge"
GeometricID(UnionFamily([OnePointDistances(), CustomFeatures([f])])).fit(X)
```

### General metric-measure spaces and graphs

The theory lives on mm-spaces, not just ℝⁿ — a precomputed distance matrix
is first-class, which is what makes the method apply to graphs:

```python
from obsdim import GeometricID, OnePointDistances
from obsdim.graphs import khop_features, shortest_path_distances

# route 1: any graph metric
D = shortest_path_distances(adjacency)
GeometricID(OnePointDistances(metric="precomputed")).fit(D)

# route 2: k-hop aggregation features of an attributed graph (TMLR Def. 5.1)
GeometricID(khop_features(adjacency, attributes, n_hops=2)).fit(None)
```

Non-uniform measures are supported everywhere via
`fit(X, sample_weight=...)`.

## Which normalization, when?

Features of different origins have codomains of different scale, and *how*
you reconcile them decides which geometric data set you measure:

- **Features share one scale** (distance functions, any metric input):
  keep them raw and normalize **globally** — `normalize="diameter"`, the
  τ·𝒟 rescaling of Tohoku §6.1.1 and the package default. This is a pure
  change of units: the mm-space itself is untouched.
- **Tabular columns with incommensurable units** (kg vs. €): scale **per
  column** at family construction — `CoordinateProjections(scale="range")`.
  This computes ∂_F for the sup-metric on min-max-scaled columns and is
  invariant under per-column affine changes of units. (Use
  `scale="global"` when the columns *do* share units — the TMLR Def. 5.1 /
  ID4GeoL convention.)
- **Unions of heterogeneous families**: scale **per member** —
  `UnionFamily([...], normalize_members=True)` divides each member by its
  own diameter, keeping each family's internal relative scales while
  making members commensurable. `FeatureScaler(family, per_feature=...)`
  offers both scalings for any family.

Anything other than one global τ changes the induced metric d_F — a
feature-engineering choice, not a change of units — so only compare
dimensions computed with the same convention. See
`examples/02_tabular_normalization.py` for a demonstration.

## Examples

Short, runnable walkthroughs live in [`examples/`](examples/):
quickstart and large-scale bounds (`01`), normalization for tabular data
(`02`), metric spaces and graphs (`03`), exchangeable feature families
(`04`).

## Related

- [scikit-dimension](https://scikit-dimension.readthedocs.io) — hub of
  classical ID estimators (MLE, TwoNN, DANCo, …) with the same
  `fit(X).dimension_` interface; `obsdim` estimators drop into its
  benchmark scripts.
- [ID4GeoL](https://github.com/mstubbemann/ID4GeoL) — the original
  experiment code for the TMLR paper (this package's test suite reproduces
  its results).
- Hille, Stubbemann, Hanika, *Reproducibility and Geometric Intrinsic
  Dimensionality* (TMLR 2024) — feature selection with this dimension, a
  downstream use case.

## Development

```bash
pip install -e .[dev]
pytest
ruff check src tests
```

The test suite verifies the implementation against exact values derived
from the papers (e.g. the nominal scale has dimension exactly n⁴, the
contranominal scale converges to 64/9 — Tohoku §6.2.2), against brute-force
subset enumeration of the defining formula, and against ID4GeoL.

## License

[AGPL-3.0-or-later](LICENSE).
