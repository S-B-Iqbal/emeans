# Changelog

All notable changes to this project are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - not yet released

First complete implementation of e-means as a scikit-learn compatible estimator
(Bessa et al., *e-means: outperforming k-means (in one-tenth the epochs)*,
arXiv:2611.09501v1).

### Added
- `EMeans` estimator: EMA centroids (Eq. 4), cosine-annealed step (Eq. 5), fitted batch
  size and peak step (Eq. 6-7, Table 2), dead-cluster revival by donor splitting
  (App. H.1), and three finalization strategies (Sec. 3.2).
- Public API: `fit`, `partial_fit`, `predict`, `transform`, `score`, `fit_predict`,
  `fit_transform`, `get_feature_names_out`; dense and CSR input; `sample_weight`.
- Compiled assignment and reduction kernels (Cython + optional OpenMP) with a NumPy
  reference backend, and an equality test suite for the two.
- `parametrize_with_checks` green for explicit and automatic hyper-parameter
  configurations; CSR, sample_weight and validation suites.
- Benchmark harness, CSV results, plots and an honest report (`benchmarks/`).

### Known limitations
- `check_sample_weight_equivalence_on_{dense,sparse}_data` are expected failures (D-041):
  mini-batch sampling plus n-dependent hyper-parameters mean an integer weight is not
  equivalent to a replicated row.  Exactness is instead covered in the `alpha0 = 1` regime.
- macOS wheels are built without OpenMP (D-053); `emeans._kernels.has_openmp()` reports it.
- Epoch budgets are integers, so the paper 0.2-epoch spacing is not reproducible (D-039).
- At laptop scale EMeans loses to a validation-swept MiniBatchKMeans on some
  configurations, and the high-`n/k`/low-`d` regimes are extrapolations of the scaling laws
  (see `benchmarks/report.md`).

[Unreleased]: https://github.com/S-B-Iqbal/emeans/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/S-B-Iqbal/emeans/releases/tag/v0.1.0
