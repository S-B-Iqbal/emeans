# SPDX-License-Identifier: BSD-3-Clause
"""Wrappers around the scikit-learn helpers emeans relies on.

Private helpers may move between releases; keeping every import here means a
broken upstream import produces one actionable message instead of a
traceback deep inside the estimator (rule 03).
"""

from __future__ import annotations

try:  # pragma: no cover - exercised at import time
    from sklearn.base import (
        BaseEstimator,
        ClassNamePrefixFeaturesOutMixin,
        ClusterMixin,
        TransformerMixin,
        _fit_context,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "emeans needs scikit-learn >= 1.6 (sklearn.base helpers); the "
        f"installed version is incompatible: {exc}"
    ) from exc

try:  # pragma: no cover - exercised at import time
    from sklearn.cluster import kmeans_plusplus
    from sklearn.utils import check_array, check_random_state
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    from sklearn.utils._param_validation import Interval, StrOptions
    from sklearn.utils.parallel import _get_threadpool_controller
    from sklearn.utils.validation import (
        _check_sample_weight,
        check_is_fitted,
        validate_data,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "emeans needs scikit-learn >= 1.6 (validation and clustering "
        f"helpers); the installed version is incompatible: {exc}"
    ) from exc

__all__ = [
    "BaseEstimator",
    "ClassNamePrefixFeaturesOutMixin",
    "ClusterMixin",
    "Interval",
    "StrOptions",
    "TransformerMixin",
    "_check_sample_weight",
    "_fit_context",
    "_get_threadpool_controller",
    "_openmp_effective_n_threads",
    "check_array",
    "check_is_fitted",
    "check_random_state",
    "kmeans_plusplus",
    "validate_data",
]
