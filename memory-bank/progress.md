# Progress

## Phase status
| Phase | Status | Approved |
|---|---|---|
| 0 Plan | done, gate pending | — |
| 1 NumPy reference | not started | — |
| 2 Compliance | not started | — |
| 3 Cython/OpenMP | not started | — |
| 4 Benchmarks | not started | — |
| 5 Packaging | not started | — |
| 6 Upstream drafts | not started | — |

## What works
- `pip install -e . --no-build-isolation` (venv on PATH) builds the editable wheel
  `emeans-0.1.0.dev0-cp310-cp310-macosx_14_0_arm64.whl` and imports `emeans`.
- `pytest --pyargs emeans -q` → `2 passed`; `pytest --doctest-modules emeans -q` → `3 passed`.
- `pre-commit run --all-files` green (ruff-check, ruff-format, hygiene hooks; cython-lint is a
  no-op until Phase 3).
- Paper of record available locally (`docs/e-means.pdf`, gitignored); Alg. 1, Eq. 5–8, Table 2,
  Sec. 3.1–3.2, App. B.1/G.2/H.1/I verified against `docs/PAPER_NOTES.md` — no discrepancies.

## Known issues / risks
- Private sklearn APIs (_fit_context, _check_sample_weight, _param_validation, _openmp_helpers,
  sklearn.utils.parallel._get_threadpool_controller) may change → `_compat.py` + nightly CI.
- sample_weight ≡ repeated rows holds exactly only in the deterministic full-batch regime
  (`partial_fit(alpha0=1.0)`), see D-024.
- Auto laws extrapolate outside DEEP's fit range (n/k < 8, n < 1e5, low d); we warn, we do not
  correct (D-021).
- CPU speedups will be well below the paper's 64-core numbers (Apple M2, 8 cores here).
- Git identity is a repo-local placeholder until the user sets their own (D-029).
