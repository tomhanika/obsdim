# Project brief: Python package for geometric intrinsic dimension

## Goal

Build a pip-installable reference implementation of the intrinsic-dimension (ID)
approach of Tom Hanika and coauthors:

- **Theory ("the Tohoku paper"):** Hanika, Schneider, Stumme,
  *Intrinsic dimension of geometric data sets*, Tohoku Mathematical Journal
  74 (2022), no. 1, 23-52. doi:10.2748/tmj.20201015a.
  Axiomatic ID for metric measure (mm-) spaces via the concentration of
  measure phenomenon (following V. Pestov): the dimension is defined through
  observable / partial diameters of push-forwards of the measure under a
  family F of 1-Lipschitz feature functions. Crucially, the dimension
  **∂_F is parametrized by the feature family F**; one-point distance
  functions d(x, ·) are the canonical instance, but F is exchangeable.
- **Scalable computation:** Stubbemann, Hanika, Schneider,
  *Intrinsic Dimension for Large-Scale Geometric Learning*, TMLR 2023.
  https://openreview.net/forum?id=85BfDdYMBY (arXiv:2210.05301).
  Sampling-based approximation (sample anchor functions instead of using all
  n distance functions) making the computation sub-quadratic.
- Existing experiment code (NOT a library): https://github.com/mstubbemann/ID4GeoL
  — use it to cross-validate numerical results; position the new package as
  the maintained, pip-installable implementation and cross-link the repos.
- Related: Hille, Stubbemann, Hanika, *Reproducibility and Geometric Intrinsic
  Dimensionality* (TMLR 2024) — uses this ID for feature selection; a good
  downstream use case for docs/examples.

The user is (or works with) the paper authors; treat the papers as the
authoritative spec. **Fetch the two papers and implement the exact functional
from their definitions** — the sketch deliberately leaves
`observable_dimension()` as a stub.

## Decisions made in conversation

### Naming
Candidate names, all verified **free on PyPI as of 2026-08-05**:
`obsdim` (recommended; from "observable diameter"), `geomdim`, `gidim`,
`concdim`, plus others. Rejected: `pestov` (named after a person),
`geodim`/geoid-like (geodesy collisions). Working name in this brief: **obsdim**
— confirm the final choice with the user before publishing, register on PyPI
early, keep module name == distribution name.

### Compatibility strategy
- **Primary interface: scikit-learn conventions.** `GeometricID(BaseEstimator)`
  with `fit(X)` and result in `.dimension_`. Rationale: ID estimation is a
  fit-once unsupervised computation, and `scikit-dimension` (skdim) — the de
  facto hub for ID estimators (MLE/Levina-Bickel, TwoNN, DANCo, ...) — uses
  exactly this pattern, so the estimator drops into existing benchmark
  scripts. Validate with `sklearn.utils.estimator_checks`.
- **No separate PyTorch/JAX backends.** Write the numerical core against the
  Python Array API standard using `array-api-compat` (as scikit-learn/SciPy
  do). Same code then runs on NumPy, PyTorch (CPU/GPU), CuPy, JAX arrays.
  Full end-to-end JAX jit-compilability is a non-goal for now (dynamic shapes
  from sampling); "accepts JAX/torch arrays and stays on device" is the goal.
- **Hard deps:** numpy, scipy, scikit-learn only. Extras:
  `obsdim[torch]` (tensor input / GPU convenience),
  `obsdim[graphs]` (networkx / PyTorch Geometric adapter that computes graph
  distances — shortest-path or diffusion — and feeds them in as precomputed).
- `metric="precomputed"` (distance matrix input) is first-class: the theory
  lives on general mm-spaces, not just R^n. This is a differentiator over
  most ID packages and what makes the method apply to graphs.

### Architecture (the key design decision — see obsdim_sketch.py)
The user emphasized that the exchangeable feature set is a major idea of the
Tohoku paper and must be first-class. Observation enabling the design: the
dimension functional never needs raw data or the metric — only the **value
matrix Phi, shape (k, n), Phi[j,i] = f_j(x_i)**, plus point weights
(the measure). Hence two decoupled layers:

1. **Core functional** `observable_dimension(phi, weights)` — push-forwards,
   partial/observable diameters over a kappa-grid, integration → dimension.
   Array-API-agnostic. Knows nothing about where features came from.
2. **`FeatureFamily` plugins** producing Phi:
   - `OnePointDistances(metric=..., anchors="all"|int)` — canonical family;
     `anchors="all"` = exact Tohoku computation (O(n^2)),
     `anchors=m` = the TMLR sampling approximation. One parameter, not two
     algorithms.
   - `CustomFeatures(funcs, lipschitz=...)` — arbitrary user callables.
   - `PrecomputedFeatures(phi)` — user hands over Phi directly (features need
     not be functions of any coordinates the package sees: kernel columns,
     model logits, graph centralities, ...).
   - `UnionFamily([...])` — F_1 ∪ ... ∪ F_m, e.g. "distances plus three
     domain features".

**Lipschitz normalization:** the axiomatics require 1-Lipschitz features.
Built-in families guarantee this by construction (distance functions,
unit-vector projections). For user functions: `lipschitz="estimate"`
(empirical constant from sampled pairs, then rescale — must respect the
chosen metric, the sketch's Euclidean shortcut is marked), a declared float,
or `None` with the documented caveat that dimensions are then only comparable
within that fixed family. Be explicit in docs about when the package computes
*the* ∂_F of the paper vs. a user-defined variant.

`sample_weight` support = non-uniform measures on the mm-space.

### Open question (ask the user, do not decide unilaterally)
Should families be allowed to be data-dependent at fit time beyond anchor
sampling (e.g., distances in an embedding learned on X itself)? Mechanically
supported by the interface, but it changes the statistical meaning of ∂_F —
may deserve an explicit flag and a documentation section.

## Suggested next steps

1. Read both papers; implement `observable_dimension` exactly per their
   definitions, with docstrings mapping each function to the corresponding
   definition/theorem ("reference implementation" credibility).
2. Package scaffold: `pyproject.toml`, `src/` layout, pytest, ruff, CI.
3. Tests:
   - reproduce known values from the papers (e.g. spheres S^n /
     Gaussian benchmarks; sqrt(n)-type concentration behavior),
   - cross-check against ID4GeoL on a small dataset,
   - invariance: `OnePointDistances(metric="precomputed")` on a Euclidean
     distance matrix == `OnePointDistances(metric="euclidean")` on raw
     points; `PrecomputedFeatures(Phi)` == the family that produced Phi,
   - `sklearn.utils.estimator_checks`,
   - Array API: same result (up to tolerance) for numpy vs torch inputs.
4. Docs: quickstart, a "choosing/creating feature families" guide (the
   package's headline feature), a graph example, comparison snippet against
   skdim estimators.

## Files in this bundle

- `CLAUDE.md` — this brief.
- `obsdim_sketch.py` — the agreed interface sketch (feature-family protocol,
  built-in families, estimator skeleton, stubbed core functional). Treat the
  *interface* as agreed with the user; the internals are illustrative.
