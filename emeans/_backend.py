# SPDX-License-Identifier: BSD-3-Clause
"""Backend switch between the NumPy reference kernels and the compiled ones.

The switch is module level (D-046): ``get_backend()``/``set_backend()`` are used by
the estimator, which keeps the resolved name in the private ``_backend``
attribute, and by the tests that compare the two implementations.  ``distances``
and ``labels_inertia`` stay on the NumPy path in Phase 3 (D-049): they run once
per fit, not once per batch.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from emeans import _numpy_backend
from emeans._kernels import HAS_KERNELS
from emeans._numpy_backend import CHUNK_SIZE

if HAS_KERNELS:  # pragma: no branch
    from emeans._kernels._assign import assign_csr as _assign_csr
    from emeans._kernels._assign import assign_dense as _assign_dense
    from emeans._kernels._reduce import reduce_csr as _reduce_csr
    from emeans._kernels._reduce import reduce_dense as _reduce_dense

_BACKEND = "cython" if HAS_KERNELS else "numpy"


def get_backend():
    """Return the active backend name, ``"numpy"`` or ``"cython"``."""
    return _BACKEND


def set_backend(name):
    """Select the backend; unknown or unavailable names raise ValueError."""
    global _BACKEND
    if name not in ("numpy", "cython"):
        raise ValueError(f"backend must be numpy or cython, got {name!r}.")
    if name == "cython" and not HAS_KERNELS:
        raise ValueError("the compiled kernels are not available in this build.")
    _BACKEND = name


def _check_threads(n_threads):
    """Threads to use: a positive int, coerced to 1 for the NumPy path."""
    n_threads = int(n_threads)
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}.")
    return 1 if _BACKEND == "numpy" else n_threads


def assign(X, centers, x_squared_norms=None, chunk_size=CHUNK_SIZE, n_threads=1):
    """Nearest-centroid assignment (Eq. 1) on the active backend."""
    if _BACKEND == "numpy":
        return _numpy_backend.assign(X, centers, x_squared_norms, chunk_size)
    n_threads = _check_threads(n_threads)
    centers64 = np.ascontiguousarray(centers, dtype=np.float64)
    if x_squared_norms is None:
        x_squared_norms = _numpy_backend.row_squared_norms(X)
    norms = np.ascontiguousarray(x_squared_norms, dtype=np.float64)
    labels = np.empty(X.shape[0], dtype=np.int32)
    if sp.issparse(X):
        matrix = X.tocsr()
        _assign_csr(
            np.ascontiguousarray(matrix.data, dtype=np.float64),
            np.ascontiguousarray(matrix.indices, dtype=np.int32),
            np.ascontiguousarray(matrix.indptr, dtype=np.int32),
            centers64,
            norms,
            labels,
            n_threads,
        )
    else:
        _assign_dense(
            np.ascontiguousarray(X, dtype=np.float64),
            centers64,
            norms,
            labels,
            n_threads,
        )
    return labels


def reduce_batch(X, labels, n_clusters, sample_weight=None, n_threads=1):
    """Weighted ``(n_j, s_j)`` of Eq. 3 on the active backend, float64."""
    if _BACKEND == "numpy":
        return _numpy_backend.reduce_batch(X, labels, n_clusters, sample_weight)
    n_threads = _check_threads(n_threads)
    weights = _numpy_backend._prepare_weights(sample_weight, X.shape[0])
    labels32 = np.ascontiguousarray(labels, dtype=np.int32)
    counts = np.zeros(n_clusters, dtype=np.float64)
    sums = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    if sp.issparse(X):
        matrix = X.tocsr()
        _reduce_csr(
            np.ascontiguousarray(matrix.data, dtype=np.float64),
            np.ascontiguousarray(matrix.indices, dtype=np.int32),
            np.ascontiguousarray(matrix.indptr, dtype=np.int32),
            labels32,
            weights,
            counts,
            sums,
            n_threads,
        )
    else:
        _reduce_dense(
            np.ascontiguousarray(X, dtype=np.float64),
            labels32,
            weights,
            counts,
            sums,
            n_threads,
        )
    return counts, sums
