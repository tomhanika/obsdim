# Development guide

`obsdim` is the reference implementation of the concentration-based
intrinsic dimension of Hanika/Schneider/Stumme (Tohoku Math. J. 74 (2022),
23–52, doi:10.2748/tmj.20201015a — "Tohoku") and the scalable algorithms of
Stubbemann/Hanika/Schneider (TMLR 2023, arXiv:2210.05301 — "TMLR"). The two
papers are the authoritative spec; every public function's docstring cites
the definition/theorem/algorithm it implements. Keep it that way.

## Layout

- `src/obsdim/_core.py` — the dimension functional. Consumes only the value
  matrix `Phi[j, i] = f_j(x_i)` plus point weights; knows nothing about
  data or metrics. Partial/observable diameters (Tohoku Def. 4.1),
  discriminability Δ = ∫ ObsDiam ∧ 1 dα (Prop. 4.5), dimension ∂ = 1/Δ²
  (Prop. 5.3); exact sliding-window algorithm (TMLR Thm 3.2 / Lemma 3.5),
  support-sequence bounds + exact refinement (TMLR §4), weighted measures.
- `src/obsdim/_families.py` — feature families producing Phi (the
  exchangeable-F layer): `OnePointDistances`, `CoordinateProjections`,
  `CustomFeatures`, `PrecomputedFeatures`, `UnionFamily`, `FeatureScaler`.
- `src/obsdim/_estimator.py` — sklearn estimator `GeometricID`
  (`fit(X).dimension_`, skdim-compatible).
- `src/obsdim/graphs.py` — shortest-path distances, TMLR Def. 5.1 k-hop
  features.
- Numerical code is written against the Python Array API (`_compat.py`);
  plain NumPy needs no extras.

## Build, test, lint

```bash
python -m venv .venv && .venv/bin/pip install -e .[dev]
.venv/bin/pytest              # full suite, seconds
.venv/bin/ruff check src tests examples
for f in examples/0*.py; do .venv/bin/python "$f"; done   # CI runs these too
```

## Testing conventions (do not "fix" these)

- Exact oracles from the papers: nominal scale ∂ = n⁴ exactly;
  contranominal scale ∂ → 64/9; Prop. 6.2 indicator partial diameters.
  The closed contranominal formula *printed* in Tohoku §6.2.2 is a
  trapezoidal approximation (n=4: paper 9/32, exact 10/32) — tests use the
  exact step-function integral on purpose; do not adjust them to the paper.
- `tests/test_id4geol_crosscheck.py` transcribes kernel functions from
  https://github.com/mstubbemann/ID4GeoL (MIT, © Stubbemann) — keep the
  attribution; obsdim must reproduce its numbers to ~1e-10.
- Brute-force subset enumeration validates the sliding window on tiny data.
- `parametrize_with_checks` keeps sklearn compatibility; run it after any
  estimator change.

## Things that look wrong but aren't

- The TMLR speedup is **support sequences over subset sizes k**, spaced
  geometrically **dense near k = n** (`s = n + 2 − geomspace(n, 2, l)`);
  dense-near-2 spacing gives ~10× worse error bounds. Anchor sampling
  (`anchors=m`) is a separate Monte-Carlo knob that biases ∂ upward
  (feature antitonicity), not the TMLR mechanism.
- Normalization defaults differ deliberately: core functions default
  `normalize=None` (the paper functional verbatim, incl. the ∧1 cap), the
  estimator defaults `normalize="diameter"` (Tohoku §6.1.1, the papers'
  experimental methodology).
- Normalization semantics: global τ = change of units; per-column
  (`CoordinateProjections(scale="range")` = sup-metric on min-max-scaled
  columns) and per-member (`UnionFamily(..., normalize_members=True)`)
  change the induced metric d_F — that is documented behavior, family
  layer only.
- Features that fail to separate points are accepted; the result equals
  the dimension of the quotient geometric data set (documented semantics).

## Releasing

1. Bump the version in `pyproject.toml` **and** `src/obsdim/__init__.py`.
2. Commit, `git tag vX.Y.Z`, push commit and tag to `github`.
3. The `Release` workflow (trusted publishing, environment `pypi`) tests,
   builds, and uploads to PyPI — no tokens. Create a GitHub release from
   the tag with `gh release create`.

Remotes: `github` (github.com/tomhanika/obsdim, primary/public) and
`origin` (GWDG GitLab, historical name `dimcon`) — push to both.

## Open questions (ask Tom, do not decide unilaterally)

- Should data-dependent families beyond anchor sampling (e.g. distances in
  an embedding learned on X itself) get an explicit flag/doc section? They
  change the statistical meaning of ∂_F.
- Optional warning when features do not separate points (quotient
  semantics) — discussed, not yet requested.
