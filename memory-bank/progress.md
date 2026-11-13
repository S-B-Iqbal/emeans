# Progress

## Phase status
| Phase | Status | Approved |
|---|---|---|
| 0 Plan | done | approved phase 0 (user, Act mode) |
| 1 NumPy reference | in progress | — |
| 2 Compliance | not started | — |
| 3 Cython/OpenMP | not started | — |
| 4 Benchmarks | not started | — |
| 5 Packaging | not started | — |
| 6 Upstream drafts | not started | — |

## What works
- P1.4: `emeans/_revival.py` - Eq. 8 splits (delta = 2**-7, D-010), dead mask at
  tau*mean(N), donors proportional to N without replacement (D-013), min(#dead, #live)
  per event (D-012), S = c*N write-back, slow state split from its own centroid
  (D-035); in-place updates, 18 tests.
- P1.3: `emeans/_numpy_backend.py` — chunked (256-row) assignment with the expanded distance form
  and lowest-index ties (D-026, D-028), `labels_inertia` in one pass, and the weighted Eq. 3
  reduction in float64 (D-016); dense **and** CSR through the same entry points, single-pass
  operations verified against literal difference-form references.
- P1.2: `emeans/_schedule.py` — `n_updates = T·⌈n/B⌉` (D-004, tail kept) and `alpha(t, M, α0, ε)`
  for `t = 1..M` (D-003; last step lands exactly on `ε·α0`), plus `alpha_schedule` for tests and
  Phase 4 plots; shared validators extracted into `emeans/_validation.py`.
- P1.1: `emeans/_scaling_laws.py` — Eq. 6/7 with the Table 2 constants (3 s.f.), log2-space rounding
  to a power of two (ties up, D-027), cap at `n`, the `/8` coefficient above 2e6, α0 computed from
  the resolved B (D-006); 29 tests, 100 % line coverage of the module.
- GitHub repository `S-B-Iqbal/emeans` (private, D-032) with `origin` configured; `main` holds the
  scaffold commit and `phase-0-plan` the Phase 0 work; labels `phase-0`…`phase-6` created.
- CI on every push: `.github/workflows/tests.yml` — wheel install and editable install jobs
  (ubuntu, py3.10); first run failed on the editable path, fixed forward in `fix(ci): keep
  editable installs non-isolated…` and green since.
- Phase-0 tracking issue: <https://github.com/S-B-Iqbal/emeans/issues/1> (gate report commented).
- `pip install -e . --no-build-isolation` (venv on PATH) builds the editable wheel
  `emeans-0.1.0.dev0-cp310-cp310-macosx_14_0_arm64.whl` and imports `emeans`.
- `pytest --pyargs emeans -q` → `2 passed`; `pytest --doctest-modules emeans -q` → `3 passed`.
- `pre-commit run --all-files` green (ruff-check, ruff-format, hygiene hooks; cython-lint is a
  no-op until Phase 3).
- Paper of record available locally (`docs/e-means.pdf`, gitignored); Alg. 1, Eq. 5–8, Table 2,
  Sec. 3.1–3.2, App. B.1/G.2/H.1/I verified against `docs/PAPER_NOTES.md` — no discrepancies.

## Known issues / risks
- meson-python editable installs performed **with** build isolation fail at import time (the
  editable loader rebuilds with the vanished overlay's `ninja`); always use
  `--no-build-isolation` after installing the build deps in the target environment.
- Private repo on the free plan has no branch protection until it is made public (D-032).
- Private sklearn APIs (_fit_context, _check_sample_weight, _param_validation, _openmp_helpers,
  sklearn.utils.parallel._get_threadpool_controller) may change → `_compat.py` + nightly CI.
- sample_weight ≡ repeated rows holds exactly only in the deterministic full-batch regime
  (`partial_fit(alpha0=1.0)`), see D-024.
- Auto laws extrapolate outside DEEP's fit range (n/k < 8, n < 1e5, low d); we warn, we do not
  correct (D-021).
- CPU speedups will be well below the paper's 64-core numbers (Apple M2, 8 cores here).
- Pushing to `origin` is pre-authorised inside this repository only (D-030); tags, releases, PyPI and
  any write on another repository still need explicit confirmation.
