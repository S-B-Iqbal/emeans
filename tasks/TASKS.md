# Tasks — tick only when the acceptance test passes

## Phase 0 — Plan
- [x] P0.1 Repo layout approved (flat, meson-python) → D-001
- [x] P0.2 Open questions Q-001…Q-026 reviewed; defaults approved or amended → D-003…D-028
- [x] P0.3 `/read-sklearn-ref.md` done; systemPatterns.md "Observed in scikit-learn" filled
- [x] P0.4 Minimal `pyproject.toml` + `meson.build` so `pip install -e .` imports `emeans`
- [x] P0.5 pre-commit installed and green on empty package
- [x] P0.6 GitHub repo `S-B-Iqbal/emeans` created (private); identity fixed; LICENSE, README.md,
      CITATION.cff added → D-030…D-034
- [x] P0.7 Minimal CI workflow (ubuntu, py3.10) green on the first push
- [x] P0.8 Phase-0 issue (#1) + phase labels created; publishing loop documented (`/publish-progress.md`)

## Phase 1 — Pure NumPy reference
- [x] P1.1 `_scaling_laws.py` + test_scaling_laws.py (Table 2, rounding, cap, n>2M)
- [x] P1.2 `_schedule.py` + test_schedule.py (Eq. 5 endpoints, monotone)
- [x] P1.3 `_numpy_backend.py` assign + weighted reduce (dense) + tests
- [x] P1.4 `_revival.py` + test_revival.py (ΣN conserved, S=c·N, Eq. 8 pattern, donor ∝ N)
- [x] P1.5 `EMeans.fit` loop (init random / k-means++ / array; EMA; shuffle per epoch)
- [x] P1.6 Finalization: last_epoch_mean / slow_ema / None + test_finalization.py
- [x] P1.7 predict / transform / score / fit_predict / fit_transform / get_feature_names_out
- [x] P1.8 partial_fit (constant α per Q-012) + test_partial_fit.py
- [x] P1.9 Determinism with random_state; float32 preservation tests
- [x] P1.10 numpydoc docstring with runnable Examples (doctest green)

## Phase 2 — Compliance
- [x] P2.1 `parametrize_with_checks` green (justified xfails only, logged in decisionLog)
- [x] P2.2 CSR support; test_sparse.py dense == sparse
- [x] P2.3 test_sample_weight.py (deterministic equivalence + statistical)
- [x] P2.4 Parameter validation messages match sklearn style

## Phase 3 — Performance
- [x] P3.1 `_kernels/_common.pxd` helpers; meson.build with openmp_dep
- [x] P3.2 `_assign.pyx` dense + CSR (chunked, prange, GEMM + argmin)
- [x] P3.3 `_reduce.pyx` dense + CSR (thread-local buffers)
- [x] P3.4 threadpoolctl BLAS limit; `_backend` switch
- [x] P3.5 test_backends.py NumPy == Cython
- [x] P3.6 Micro-benchmark: Cython vs NumPy path speedup recorded in progress.md

## Phase 4 — Benchmarks
- [x] P4.1 `benchmarks/datasets.py` (blobs, low-d, high n/k, 1–2 embedding sets) with held-out split
- [x] P4.2 epochs-to-Lloyd@25 (0.2-epoch steps), ≥5 seeds
- [x] P4.3 test MSE vs wall-clock plots; MiniBatchKMeans swept on validation
- [x] P4.4 `benchmarks/report.md` incl. hardware, versions, losing cases

## Phase 5 — Packaging & GitHub
- [x] P5.1 README (citation, BibTeX, not-affiliated note), LICENSE (BSD-3), CITATION.cff, CHANGELOG
- [x] P5.2 Sphinx + numpydoc + sphinx-gallery example; docs build
- [x] P5.3 GitHub Actions: tests matrix, wheels (cibuildwheel), docs, nightly-sklearn
- [x] P5.4 Exact gh/git commands printed — NOT executed

## Phase 6 — Upstream drafts (saved in docs/upstream/, never posted)
- [ ] P6.1 scikit-learn-contrib inclusion request
- [ ] P6.2 Comment for scikit-learn issue #35383
- [ ] P6.3 Discussion text on MiniBatchKMeans improvements
- [ ] P6.4 MiniBatchKMeans PR sketch (files, params, back-compat, tests)
