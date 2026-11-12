---
paths:
  - "emeans/_kernels/**"
  - "**/meson.build"
  - "pyproject.toml"
---
# 07 — Cython / OpenMP / build rules (Phase 3+)

- Build backend: meson-python (match scikit-learn). Every `py.extension_module` lists
  `dependencies: [np_dep, openmp_dep]` — scikit-learn once shipped modules without OpenMP by omission.
- Add an import-time sanity check that the compiled module reports OpenMP support consistent with the build.
- Directives: `# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False`.
- Use the `floating` fused type so float32 and float64 compile from one source.
- Assignment kernel: mirror the chunked pattern of `_k_means_lloyd.pyx` — CHUNK_SIZE = 256 rows,
  `prange(n_chunks, schedule="static", nogil=True, num_threads=n_threads)`, GEMM for x·cᵀ plus
  precomputed ‖c‖², argmin per row. Separate dense and CSR functions.
- Reduction kernel: per-thread local (k,) counts and (k, d) sums, then a final reduction
  (mirror `_k_means_minibatch.pyx` / lloyd's local buffers). No atomics on the hot path.
- Respect `threadpoolctl`: limit BLAS to 1 thread around OpenMP regions.
- Cannot reliably cimport sklearn's `.pxd` across packages: re-implement small helpers in
  `emeans/_kernels/_common.pxd` with a citation of the sklearn file they mirror.
- Lint with cython-lint. Generated `.c` files are never committed.
- Equality vs NumPy backend is the acceptance test, not speed.
