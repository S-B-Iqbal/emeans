# System patterns

## Architecture
- `EMeans` (estimator) in `emeans/_emeans.py` orchestrates; math lives in pure helpers:
  `_scaling_laws.py` (Eq. 6–7), `_schedule.py` (Eq. 5), `_revival.py` (App. H.1),
  `_numpy_backend.py` (assignment + weighted count/sum reduction).
- Backend switch `_backend in {"numpy", "cython"}`; NumPy path is the oracle.
- State: `_counts` (k,), `_sums` (k, d) dense even for CSR input; centroids derived `S / N`.

## Training loop (one fit)
init centroids → N=1e-10, S=N·c → for epoch in T: perm = rng.permutation(n) →
for batch: α(t) → assign → reduce (n_j, s_j) → EMA update → [every revival_period] revive →
[last epoch] record labels → finalize → labels_/inertia_ (see Q-016) → resolved attrs.

## Kernel pattern (Phase 3, mirrors sklearn lloyd)
Chunks of 256 rows; per chunk: GEMM x·cᵀ, add ‖c‖², argmin; thread-local count/sum buffers;
final reduction; BLAS limited to 1 thread around OpenMP.

## Contrast with MiniBatchKMeans reassignment (for Phase 6 pitch)
sklearn: centers with weight_sum < reassignment_ratio·max are reset to random batch samples
(≤ half the batch), counts reset to the min live count. e-means: split a donor ∝ N (App. H.1).

## Observed in scikit-learn @<commit>
(Fill in via /read-sklearn-ref.md before Phase 1.)
