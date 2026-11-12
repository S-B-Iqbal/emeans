# Project brief — emeans

Build `emeans`, a scikit-learn-compatible implementation of e-means (arXiv:2611.09501v1,
Bessa et al., FAIR at Meta, 6 Nov 2026), as estimator `EMeans`.

## Must
- Pass `check_estimator` / `parametrize_with_checks`.
- Dense float32/float64 (dtype preserved) and CSR input; sample_weight; partial_fit.
- Pure-NumPy reference + Cython/OpenMP kernels with equality tests.
- Honest CPU benchmarks vs KMeans (lloyd, 25 it) and MiniBatchKMeans.
- BSD-3; GitHub + PyPI; scikit-learn-contrib proposal; upstream discussion (no estimator PR).

## Definition of done
`pip install -e .` works · pytest + check_estimator pass · CI green (Linux/macOS/Windows,
Python 3.10–3.13, cibuildwheel wheels) · benchmarks reproduce the qualitative result
(EMeans reaches KMeans@25 test MSE in ~2–3 epochs) or explain why not · docs build.

## Hard constraints
Implement from the paper only · never copy Meta's code · never invent paper details ·
phase gates with explicit user approval · never push/publish without confirmation.
