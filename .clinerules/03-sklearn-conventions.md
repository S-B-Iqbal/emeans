# 03 — scikit-learn conventions (must follow)

## Class
```python
class EMeans(ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator):
```
Mixins left, `BaseEstimator` right. `__init__` stores every parameter verbatim as an attribute of
the same name — no validation, no defaults resolution, no derived state, no logic.

## Validation
- `_parameter_constraints: dict` with `Interval`, `StrOptions` (from `sklearn.utils._param_validation`),
  `"array-like"`, `"random_state"`, `"verbose"`, `None`, `Integral`, `Real`.
- Decorate `fit` and `partial_fit` with `@_fit_context(prefer_skip_nested_validation=True)`.
- Input:
  `X = validate_data(self, X, accept_sparse="csr", dtype=[np.float64, np.float32], order="C",
  accept_large_sparse=False, copy=False, reset=<True in fit / first partial_fit, else False>)`.
- `sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)`.
- `random_state = check_random_state(self.random_state)` once per fit; derive all randomness from it.
- `check_is_fitted(self)` in predict/transform/score.
- Raise `ValueError` if `n_samples < n_clusters`, using the same message style as `KMeans`.
- Tags via `__sklearn_tags__` (set `input_tags.sparse = True`); never `_more_tags`.

## Public API parity with MiniBatchKMeans
`fit(X, y=None, sample_weight=None)`, `partial_fit(X, y=None, sample_weight=None)`, `predict`,
`fit_predict`, `transform`, `fit_transform`, `score` (= −inertia on X), `get_feature_names_out`.

## Fitted attributes
`cluster_centers_`, `labels_`, `inertia_`, `n_iter_` (epochs completed), `n_steps_`
(EMA updates performed — same meaning as MiniBatchKMeans: minibatches processed),
`n_features_in_`, `feature_names_in_`, `batch_size_` (resolved), `alpha0_` (resolved).
Private state (if kept): `_counts` (N), `_sums` (S), `_slow_counts`, `_slow_sums`.

## Mirror these names from `sklearn/cluster/_kmeans.py` where semantics match
`_labels_inertia`, `_labels_inertia_threadpool_limit`, `_check_params_vs_input`,
`_init_centroids`, `_check_test_data`, `_n_threads`. Verify exact current signatures in
`_ref/scikit-learn` — do not trust memory.

## Threads
`self._n_threads = _openmp_effective_n_threads()`; run assignment loops with BLAS limited to
1 thread (threadpoolctl / sklearn's threadpool controller helpers), as KMeans does, to avoid
nested oversubscription.

## Dependency floor
`scikit-learn>=1.6` (public `validate_data`, `__sklearn_tags__`). Private helpers
(`_fit_context`, `_check_sample_weight`, `_param_validation`, `_openmp_helpers`) may change:
wrap imports in `emeans/_compat.py` and run a nightly-scikit-learn CI job.

## Docstrings
numpydoc, sections: Parameters, Attributes, See Also, Notes, References, Examples.
The Examples section must run as a doctest (`pytest --doctest-modules emeans`).
