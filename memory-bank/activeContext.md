# Active context


**Phase:** 3 - Cython/OpenMP kernels - **Mode:** ACT (approved with Phase 2)
**Status:** Phase 3 tasks P3.1-P3.6 done on `phase-3-kernels`; gate pending.
Waiting for `approved phase 3` before merging to `main`.

## Phase 2 (done)
- `parametrize_with_checks` green (223 passed / 4 justified xfails, D-041); two real
  bugs fixed (D-042, D-043); dense == CSR, sample_weight and validation suites added.

## Phase 1 (done)
107 unit tests, 109 doctests, 99% coverage, CI green (wheel + editable jobs).
Modules: `_scaling_laws` (Eq. 6-7), `_schedule` (Eq. 5), `_numpy_backend`
(Eq. 1/3, dense + CSR), `_revival` (App. H.1), `_compat`, `_validation`,
`_emeans` (`EMeans`: fit, finalization, public API, partial_fit).

## Phase 3 plan (approved)
Issue #4, branch `phase-3-kernels`. OpenMP is optional (D-044): Apple clang has
no `-fopenmp`, so the local build is serial while CI exercises the parallel path.

- P3.1 `emeans/_kernels/`: `_common.pxd` (helpers citing sklearn `_k_means_common.pxd`),
  `meson.build` with `np_dep` + `openmp_dep`; root `meson.build` gains the Cython
  generator; build proof with a dummy extension.
- P3.2 `_assign.pyx`: `floating` fused type, CHUNK_SIZE = 256, chunked
  `prange(nogil=True, num_threads=n_threads)`, `dgemm` (from
  `scipy.linalg.cython_blas`) + `||x||^2 + ||c||^2` + argmin with strict `<`;
  dense and CSR variants.
- P3.3 `_reduce.pyx`: per-thread `(k,)` counts and `(k, d)` sums, final float64
  reduction, dense and CSR.
- P3.4 `emeans/_backend.py`: `get_backend`/`set_backend` dispatching assign, reduce,
  distances, labels_inertia; the estimator keeps a private `self._backend`; the
  training loop is wrapped in `_get_threadpool_controller().limit(...)` (D-038).
- P3.5 `test_backends.py`: NumPy == Cython (identical labels, allclose counts/sums)
  over dense/CSR x float32/float64.
- P3.6 micro-benchmark recorded in `progress.md` (serial locally, OpenMP in CI).

Decisions: D-044 (optional OpenMP), D-045 (fused type but float64 dispatch),
D-046 (module-level switch, no new public parameter).

## Publishing (D-030…D-034)
- `origin` = `S-B-Iqbal/emeans` (private until the Phase 1 gate, then public — D-032); loop defined
  in `.clinerules/workflows/publish-progress.md`: push after every green task, tick the phase issue,
  merge to `main` only after a gate is approved.
- CI: `.github/workflows/tests.yml` (ubuntu, py3.10) on every push (D-034); full matrix in Phase 5.
  Green on `fix(ci): keep editable installs non-isolated…` (the very first run exposed the
  meson-python editable/build-isolation trap).
- Phase 0.6 (repo creation, LICENSE/README/CITATION, CI) is tracked as P0.6–P0.8 in `TASKS.md` and
  in issue #1.

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"` where
  the resolved batch size would be large.
- Repo-local git identity is `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031) and
  pushing to `origin` is pre-authorised (D-030); tags, releases, PyPI and other repositories are not.
