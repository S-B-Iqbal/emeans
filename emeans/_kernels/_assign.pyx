# SPDX-License-Identifier: BSD-3-Clause
"""Chunked nearest-centroid assignment (Eq. 1), dense and CSR.

The structure mirrors ``sklearn/cluster/_k_means_lloyd.pyx``: per-thread buffers
allocated inside ``with nogil, parallel(num_threads=n_threads)``, a
``prange(n_chunks, schedule="static")`` loop, one GEMM per chunk and an argmin
with a strict ``<`` so ties resolve to the lowest cluster index (D-028).  The
GEMM call is derived from the BLAS convention rather than copied: for row-major
``X (m, d)`` and ``C (k, d)`` the call is ``dgemm("T", "N", k, m, d, -2, C, d,
X, d, 0, dist, k)``, because both buffers are the transposes of the Fortran
views BLAS expects and the row-major ``(m, k)`` result is the Fortran
``(k, m)`` one, so a single call computes ``dist = X @ C.T``.
"""

from cython cimport floating
from cython.parallel cimport parallel, prange
from libc.stdlib cimport free, malloc

from scipy.linalg.cython_blas cimport dgemm, sgemm

from emeans._kernels._common import CHUNK_SIZE
from emeans._kernels._common cimport _sq_norm

import numpy as np

ctypedef int blas_int


cdef inline void _gemm_distances(int m, int n_clusters, int n_features,
                                 floating alpha, const floating* X_chunk,
                                 const floating* C,
                                 floating* dist) noexcept nogil:
    """``dist = alpha * X_chunk @ C.T`` for row-major inputs."""
    cdef:
        char trans_a = 84  # BLAS trans flag T
        char trans_b = 78  # BLAS trans flag N
        blas_int k_c = n_clusters
        blas_int rows = m
        blas_int inner = n_features
        blas_int ld_c = n_features
        blas_int ld_x = n_features
        blas_int ld_d = n_clusters
        floating alpha_ = alpha
        floating beta = 0.0
    if floating is float:
        sgemm(&trans_a, &trans_b, &k_c, &rows, &inner, &alpha_, <float*> C,
              &ld_c, <float*> X_chunk, &ld_x, &beta, <float*> dist, &ld_d)
    else:
        dgemm(&trans_a, &trans_b, &k_c, &rows, &inner, &alpha_, <double*> C,
              &ld_c, <double*> X_chunk, &ld_x, &beta, <double*> dist, &ld_d)


def assign_dense(const floating[:, ::1] X, const floating[:, ::1] centers,
                 const double[::1] x_squared_norms,
                 int[::1] labels, int n_threads):
    """Fill ``labels`` with ``argmin_j ||x_i - c_j||**2`` (Eq. 1), dense X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating, C order
    centers : ndarray of shape (n_clusters, n_features), dtype=floating
    x_squared_norms : ndarray of shape (n_samples,), dtype float64
    labels : ndarray of shape (n_samples,), dtype int32, written in place
    n_threads : int
        OpenMP threads to use (1 builds and runs serial, D-044).
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers.shape[0]
        int chunk_size = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks, chunk_idx, start, end, n_chunk, i, j
        int best_index
        double x2, best, candidate
        floating alpha = -2.0
        double[::1] centers_sq
        floating* dist

    if n_samples == 0:
        return

    centers_sq = np.empty(n_clusters, dtype=np.float64)
    for j in range(n_clusters):
        centers_sq[j] = _sq_norm(centers[j])

    n_chunks = n_samples // chunk_size
    if n_chunks * chunk_size != n_samples:
        n_chunks += 1

    with nogil, parallel(num_threads=n_threads):
        dist = <floating*> malloc(chunk_size * n_clusters * sizeof(floating))
        if dist == NULL:
            with gil:
                raise MemoryError("could not allocate the distance chunk")
        for chunk_idx in prange(n_chunks, schedule="static"):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            if end > n_samples:
                end = n_samples
            n_chunk = end - start
            _gemm_distances(n_chunk, n_clusters, n_features, alpha,
                            &X[start, 0], &centers[0, 0], dist)
            for i in range(n_chunk):
                x2 = x_squared_norms[start + i]
                best_index = 0
                best = dist[i * n_clusters] + x2 + centers_sq[0]
                for j in range(1, n_clusters):
                    candidate = dist[i * n_clusters + j] + x2 + centers_sq[j]
                    if candidate < best:
                        best = candidate
                        best_index = j
                labels[start + i] = best_index
        free(dist)


def assign_csr(const floating[::1] data, const int[::1] indices,
               const int[::1] indptr, const floating[:, ::1] centers,
               const double[::1] x_squared_norms,
               int[::1] labels, int n_threads):
    """Same as :func:`assign_dense` for CSR input (no GEMM: rows are sparse).

    ``data``/``indices``/``indptr`` are the CSR arrays with int32 indices; the
    per-row dot products accumulate over the non-zeros of the row.
    """
    cdef:
        int n_samples = indptr.shape[0] - 1
        int n_clusters = centers.shape[0]
        int i, j, p, best_index
        double dot, best, candidate
        double[::1] centers_sq

    centers_sq = np.empty(n_clusters, dtype=np.float64)
    for j in range(n_clusters):
        centers_sq[j] = _sq_norm(centers[j])

    with nogil, parallel(num_threads=n_threads):
        for i in prange(n_samples, schedule="static"):
            best_index = 0
            best = 0.0
            for j in range(n_clusters):
                dot = 0.0
                for p in range(indptr[i], indptr[i + 1]):
                    dot = dot + data[p] * centers[j, indices[p]]
                candidate = x_squared_norms[i] + centers_sq[j] - 2.0 * dot
                if j == 0 or candidate < best:
                    best = candidate
                    best_index = j
            labels[i] = best_index
