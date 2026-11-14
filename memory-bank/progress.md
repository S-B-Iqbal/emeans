# Progress

## Phase status
| Phase | Status | Approved |
|---|---|---|
| 0 Plan | done | approved phase 0 (user, Act mode) |
| 1 NumPy reference | done | approved phase 1 (user, Act mode) |
| 2 Compliance | done | approved phase 2 (user, Act mode) |
| 3 Cython/OpenMP | done | approved phase 3 (user) |
| 4 Benchmarks | done | approved phase 4 (user) |
| 5 Packaging | done | approved phase 5 (user, option A `emeans-cluster`) |
| 6 Upstream drafts | done, gate pending | — |

## What works
- P6.0-P6.5: six drafts under `docs/upstream/` (parity matrix vs paper/`sklearn-contrib-emeans` 0.3.1/Faiss;
  contrib inclusion request with the four gaps stated; a <=350-word comment for scikit-learn #35383
  quoting our serial-CPU numbers *including* the losses; a MiniBatchKMeans improvement discussion
  written against the real code in `_ref/scikit-learn`; a PR sketch (API, files, tests, back-compat);
  and a user-example skeleton for embedding clustering, which is what `betatim` asked for).
  **Nothing posted**: rule 01/10. Four open questions Q-030-Q-033 (fractional `n_epochs`, naming,
  offering `sample_weight`/`partial_fit` upstream, OpenMP wheels) are filed with proposed defaults.
- P5.1-P5.4: distribution renamed to **emeans-cluster** (D-052, import stays `emeans`), CHANGELOG + README with the benchmark table, Sphinx/numpydoc/sphinx-gallery docs (`sphinx -W -n` green), CI matrix 3 OS x Python 3.10-3.13 + docs + coverage + nightly scikit-learn + cibuildwheel artefacts, and `docs/release-runbook.md` (printed only). Two CI defects fixed: gallery sources were gitignored, and MSVC OpenMP linked a missing vcomp DLL (D-056: Windows builds serial now).
- P4.1-P4.4: `benchmarks/` suite (datasets, harness, runner, plots, report). EMeans reaches the Lloyd@25 test MSE in **2-4 epochs** on the blob datasets (-3.2% at d=32, -19.8% at d=96, median of 5 seeds) but **loses to a validation-swept MiniBatchKMeans on 3 of 4 datasets and to Lloyd on wall-clock (~2x slower serial); the high-n/k and low-d configurations are documented losses. Raw CSV + 3 PNGs + `report.md` committed.
- P3.1-P3.6: compiled kernels live. `_assign.pyx` (chunked OpenMP assignment, dense + CSR) and `_reduce.pyx` (thread-local weighted reduction) verify bit-exactly against the NumPy oracle; `_backend.py` dispatches to Cython by default (D-051), BLAS limited to one thread (closes D-038). OpenMP is optional (D-044): local builds are serial, CI reports `True 4`. Micro-benchmark (n=20000, k=64, d=32, 1 epoch, 40 updates): NumPy 0.050 s vs Cython 0.015 s -> **3.3x serial**, identical inertia. 269 tests, 4 xfailed.
- P2.1-P2.4: `parametrize_with_checks` green for explicit and auto configs (223 passed / 4 justified xfails, D-041); dense == CSR equivalence; sample_weight exactness in the alpha0 = 1 regime plus a statistical repeated-rows test; validation and warning surface pinned. Two real bugs fixed (D-042, D-043). 248 passed, 4 xfailed, 99% coverage.
- P1.7-P1.10: public API (`predict`, `transform` in Euclidean distances, `score`, `fit_predict`, `fit_transform`, `get_feature_names_out`), streaming `partial_fit` with a frozen constant step (D-014), float32/float64 agreement tests, and the full numpydoc docstring with runnable doctests. 107 unit tests, 109 doctests, 99% coverage, CI green.
- P1.5/P1.6: `emeans/_compat.py` and `emeans/_emeans.py` - `EMeans.fit` runs the EMA loop (Eq. 2, global t, D-023), revives every `revival_period` updates, resolves `batch_size_`/`alpha0_` (Eq. 6-7), supports init random/k-means++/array, and finalises with last_epoch_mean / slow_ema / None (D-008, D-009); the extra assignment pass (D-018) sets labels_/inertia_. 90 tests, 99% coverage, _emeans.py 100%.
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
- GitHub repository `S-B-Iqbal/emeans` (**public** since the Phase-1 gate, D-032) with `origin` configured; `main` holds the
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
