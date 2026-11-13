# SPDX-License-Identifier: BSD-3-Clause
"""Weighted per-cluster count and sum (Eq. 3), dense and CSR.

Each thread accumulates into its own ``(n_clusters,)`` count buffer and
``(n_clusters, n_features)`` sum buffer (rule 07), which are reduced into the
float64 outputs afterwards (D-016).  The number of threads is capped so the
thread-local buffers stay below 64 MB (D-047); rows are split into chunks of
``CHUNK_SIZE`` like the assignment kernel.
"""

from cython cimport floating
from cython.parallel cimport parallel, prange, threadid

from emeans._kernels._common import CHUNK_SIZE

import numpy as np

# bytes of thread-local buffers we allow (64 MB, D-047)
cdef long long MAX_LOCAL_BYTES = 67108864


cdef inline int _thread_budget(int n_clusters, int n_features) noexcept nogil:
    """Largest number of threads whose local buffers fit in the budget."""
    cdef long long per_thread = 8 * (<long long> n_clusters) * (n_features + 1)
    if per_thread <= 0:
        return 1
    cdef long long budget = MAX_LOCAL_BYTES // per_thread
    if budget < 1:
        return 1
    return <int> budget


def reduce_dense(const floating[:, ::1] X, const int[::1] labels,
                 const double[::1] weights, double[::1] counts,
                 double[:, ::1] sums, int n_threads):
    """Accumulate ``n_j`` and ``s_j`` of Eq. 3 for a dense batch, in float64.

    ``counts`` and ``sums`` are accumulated into (the caller zeroes them).
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = counts.shape[0]
        int chunk_size = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks, chunk_idx, start, end, i, f, j, tid, t
        int budget = _thread_budget(n_clusters, n_features)
        double w
        double[:, ::1] local_sums
        double[::1] local_counts
        double* buf_counts
        double* buf_sums

    n_threads = min(n_threads, budget)
    if n_threads < 1:
        n_threads = 1
    if n_samples == 0:
        return
    local_counts = np.zeros(n_threads * n_clusters, dtype=np.float64)
    local_sums = np.zeros((n_threads * n_clusters, n_features), dtype=np.float64)

    n_chunks = n_samples // chunk_size
    if n_chunks * chunk_size != n_samples:
        n_chunks += 1

    with nogil, parallel(num_threads=n_threads):
        tid = threadid()
        buf_counts = &local_counts[tid * n_clusters]
        buf_sums = &local_sums[tid * n_clusters, 0]
        for chunk_idx in prange(n_chunks, schedule="static"):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            if end > n_samples:
                end = n_samples
            for i in range(start, end):
                j = labels[i]
                w = weights[i]
                buf_counts[j] = buf_counts[j] + w
                for f in range(n_features):
                    buf_sums[j * n_features + f] = (
                        buf_sums[j * n_features + f] + w * X[i, f])

    for t in range(n_threads):
        for j in range(n_clusters):
            counts[j] = counts[j] + local_counts[t * n_clusters + j]
            for f in range(n_features):
                sums[j, f] = sums[j, f] + local_sums[t * n_clusters + j, f]


def reduce_csr(const floating[::1] data, const int[::1] indices,
               const int[::1] indptr, const int[::1] labels,
               const double[::1] weights, double[::1] counts,
               double[:, ::1] sums, int n_threads):
    """Same as :func:`reduce_dense` for CSR input."""
    cdef:
        int n_samples = indptr.shape[0] - 1
        int n_features = sums.shape[1]
        int n_clusters = counts.shape[0]
        int i, p, f, j, tid, t
        int budget = _thread_budget(n_clusters, n_features)
        double w
        double[:, ::1] local_sums
        double[::1] local_counts
        double* buf_counts
        double* buf_sums

    n_threads = min(n_threads, budget)
    if n_threads < 1:
        n_threads = 1
    if n_samples == 0:
        return
    local_counts = np.zeros(n_threads * n_clusters, dtype=np.float64)
    local_sums = np.zeros((n_threads * n_clusters, n_features), dtype=np.float64)

    with nogil, parallel(num_threads=n_threads):
        tid = threadid()
        buf_counts = &local_counts[tid * n_clusters]
        buf_sums = &local_sums[tid * n_clusters, 0]
        for i in prange(n_samples, schedule="static"):
            j = labels[i]
            w = weights[i]
            buf_counts[j] = buf_counts[j] + w
            for p in range(indptr[i], indptr[i + 1]):
                f = indices[p]
                buf_sums[j * n_features + f] = (
                    buf_sums[j * n_features + f] + w * data[p])

    for t in range(n_threads):
        for j in range(n_clusters):
            counts[j] = counts[j] + local_counts[t * n_clusters + j]
            for f in range(n_features):
                sums[j, f] = sums[j, f] + local_sums[t * n_clusters + j, f]
