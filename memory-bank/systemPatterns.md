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

## Observed in scikit-learn @7b2b14e0a89eedfa514dc9885e35a0ecade20cd0

Source: `_ref/scikit-learn` (main branch, requires Python >= 3.12 — structure reference only;
the behavioural reference is the installed scikit-learn 1.7.2 wheel).
Files read: `sklearn/cluster/_kmeans.py`, `sklearn/cluster/meson.build`, `meson.build`.

### Validation and plumbing
- `_parameter_constraints` style: `{"n_clusters": [Interval(Integral, 1, None, closed="left")],
  "init": [StrOptions({"k-means++", "random"}), callable, "array-like"], "verbose": ["verbose"],
  "random_state": ["random_state"]}`.
- `fit` / `partial_fit` decorated with `@_fit_context(prefer_skip_nested_validation=True)`
  (imported from `sklearn.base`).
- `validate_data(self, X, accept_sparse="csr", dtype=[np.float64, np.float32], order="C",
  accept_large_sparse=False)`; `_check_test_data` adds `reset=False`.
- `partial_fit`: `has_centers = hasattr(self, "cluster_centers_")`,
  `validate_data(..., reset=not has_centers)`, and the RNG survives across calls:
  `self._random_state = getattr(self, "_random_state", check_random_state(self.random_state))`.
- `sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)`; an array `init` is
  validated with `check_array(init, dtype=X.dtype, copy=True, order="C")` +
  `_validate_center_shape`.
- `self._n_threads = _openmp_effective_n_threads()` in `fit` / first `partial_fit`;
  `_check_mkl_vcomp` emits the vcomp+MKL warning.
- Warning style: `warnings.warn(msg, RuntimeWarning, stacklevel=2)`.

### Threads / BLAS
- Around the training loop:
  `with _get_threadpool_controller().limit(limits="sequential_blas_under_openmp"):`
- Module level:
  `_labels_inertia_threadpool_limit = _threadpool_controller_decorator(
  limits="sequential_blas_under_openmp")(_labels_inertia)` — both from `sklearn.utils.parallel`
  (this supersedes the older direct `threadpoolctl.threadpool_limits` usage).

### Error / warning messages to mirror
- `ValueError(f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}.")`
- `"The shape of the initial centers {centers.shape} does not match the number of clusters {k}."`
  / `"... does not match the number of features of the data {d}."`

### Initialization
- `init="random"`:
  `random_state.choice(n_samples, size=n_clusters, replace=False, p=sample_weight / sample_weight.sum())`
  → distinct rows, probability proportional to `sample_weight` (uniform when `None`). Basis of D-026.
- `init="k-means++"` → `_kmeans_plusplus(X, n_clusters, random_state=..., x_squared_norms=...,
  sample_weight=..., n_threads=...)`; the public
  `kmeans_plusplus(X, n_clusters, *, sample_weight=None, x_squared_norms=None,
  random_state=None, n_local_trials=None)` is wrapped in
  `@validate_params(..., prefer_skip_nested_validation=True)`.

### Public methods and attributes
- `predict`: `check_is_fitted` → `_check_test_data` → `np.ones(n, dtype=X.dtype)` weights →
  `_labels_inertia_threadpool_limit(..., return_inertia=False)`.
- `transform` → `euclidean_distances(X, cluster_centers_)` (true distances, not squared).
- `score` → `-inertia` from `_labels_inertia_threadpool_limit`.
- `fit_predict` → `self.fit(X, sample_weight=sample_weight).labels_`.
- `__sklearn_tags__`: `tags = super().__sklearn_tags__(); tags.input_tags.sparse = True`.
- Fitted: `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, `n_steps_`, `_n_features_out`
  (`= cluster_centers_.shape[0]`, used by `get_feature_names_out`).
- MiniBatchKMeans: `_batch_size = min(self.batch_size, X.shape[0])`,
  `_counts = np.zeros(n_clusters, dtype=X.dtype)`, `n_steps_ = i + 1`,
  `n_iter_ = int(np.ceil(((i + 1) * _batch_size) / n_samples))`; with `compute_labels=True`
  it runs exactly one extra `_labels_inertia_threadpool_limit(X, sample_weight,
  cluster_centers_)` after training (the pattern adopted in D-018); `partial_fit` increments
  `n_steps_` by 1 per call and refreshes `labels_` / `inertia_` from the current batch.

### Build
- Root `meson.build`: `project('scikit-learn', 'c', 'cpp', 'cython', version: ...,
  meson_version: '>= 1.9.0')`, `py = import('python').find_installation(pure: false)`,
  then `install_subdir('sklearn', install_dir: py.get_install_dir())`.
- Extensions (`sklearn/cluster/meson.build`): `py.extension_module(name, [sources, ...],
  dependencies: [np_dep, openmp_dep], subdir: 'sklearn/cluster', install: true)` inside a
  `foreach` over a metadata dict — the pattern Phase 3 will mirror with `emeans/_kernels`.

### Differences that matter for emeans
- sklearn's `_mini_batch_step` is MacQueen-style with `reassignment_ratio` random resets;
  e-means uses EMA statistics plus split-based revival (App. H.1). No code is shared.
- `n_iter_` means different things (implied epochs vs. real epochs, D-019).
- The `_kmeans_plusplus` Cython kernels are private and cannot be cimported across packages;
  Phase 3 re-implements chunked assignment in `emeans/_kernels` with a citation comment.
