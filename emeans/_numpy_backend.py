# SPDX-License-Identifier: BSD-3-Clause
"""Reference (NumPy) kernels: assignment (Eq. 1) and per-batch reduction (Eq. 3).

This module is the oracle the Cython kernels of Phase 3 must reproduce, so it favours
accuracy and readability over speed:

* squared distances use the expanded form ``||x||**2 - 2 x.c + ||c||**2`` with float64
  accumulators, and ties resolve to the lowest cluster index (D-026, D-028);
* ``assign`` and ``labels_inertia`` walk samples in chunks of :data:`CHUNK_SIZE`, which
  keeps the ``(chunk, k)`` distance buffer bounded and mirrors the structure of Phase 3;
* the reduction accumulates float64 counts and sums for the EMA state (D-016).

Both dense and CSR inputs are supported by the same entry points; the centroids and the
returned statistics are always dense.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from emeans._validation import check_positive_integer

# Same chunk size as scikit-learn's chunked Lloyd/mini-batch kernels
# (sklearn/cluster/_k_means_common.pxd); Phase 3 mirrors that layout.
CHUNK_SIZE = 256


def row_squared_norms(X) -> np.ndarray:
    """Squared Euclidean norm of every row of ``X``, as ``float64``.

    ``X`` may be a dense array or a CSR matrix; the result is dense and is meant to be
    computed once per fit (the training set does not change).
    """
    if sp.issparse(X):
        norms = X.multiply(X).sum(axis=1)
        return np.asarray(norms, dtype=np.float64).ravel()
    X64 = _as_dense_float64(X)
    return np.einsum("ij,ij->i", X64, X64)


def assign(X, centers, x_squared_norms=None, chunk_size: int = CHUNK_SIZE):
    """Assign every row of ``X`` to its nearest centroid (Eq. 1).

    ``a(x) = argmin_j ||x - c_j||**2``, with ties going to the lowest ``j`` (D-028).

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Samples to assign; a CSR matrix is accepted.
    centers : ndarray of shape (n_clusters, n_features)
        Centroids, dense, in any float dtype.
    x_squared_norms : ndarray of shape (n_samples,), default=None
        Row norms of ``X``, if already computed (see :func:`row_squared_norms`).
    chunk_size : int, default=256
        Number of rows processed at once.

    Returns
    -------
    labels : ndarray of shape (n_samples,), dtype int32
        Index of the nearest centroid for each row.
    """
    centers64, centers_sq, x_squared_norms = _prepare_centers(
        centers, X, x_squared_norms
    )
    chunk_size = check_positive_integer("chunk_size", chunk_size)

    n_samples = X.shape[0]
    labels = np.empty(n_samples, dtype=np.int32)
    for start, stop in _chunks(n_samples, chunk_size):
        distances = _distances(
            X[start:stop], centers64, centers_sq, x_squared_norms[start:stop]
        )
        labels[start:stop] = np.argmin(distances, axis=1)
    return labels


def labels_inertia(X, centers, sample_weight=None, chunk_size: int = CHUNK_SIZE):
    """Labels of Eq. 1 and the weighted inertia, in a single pass.

    Returns
    -------
    labels : ndarray of shape (n_samples,), dtype int32
    inertia : float
        ``sum_i w_i * ||x_i - c_{a(i)}||**2``, accumulated in float64.
    """
    centers64, centers_sq, x_squared_norms = _prepare_centers(centers, X, None)
    chunk_size = check_positive_integer("chunk_size", chunk_size)
    weights = _prepare_weights(sample_weight, X.shape[0])

    n_samples = X.shape[0]
    labels = np.empty(n_samples, dtype=np.int32)
    inertia = 0.0
    for start, stop in _chunks(n_samples, chunk_size):
        distances = _distances(
            X[start:stop], centers64, centers_sq, x_squared_norms[start:stop]
        )
        labels[start:stop] = np.argmin(distances, axis=1)
        inertia += float(np.dot(weights[start:stop], distances.min(axis=1)))
    return labels, inertia


def reduce_batch(X, labels, n_clusters: int, sample_weight=None):
    """Weighted count and sum that a batch contributes to every cluster (Eq. 3).

    ``n_j = sum_{a(x)=j} w(x)`` and ``s_j = sum_{a(x)=j} w(x) * x``, both float64 in
    the EMA state (D-016).

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The batch, dense or CSR.
    labels : array-like of shape (n_samples,)
        Assignment of each row of ``X``, as returned by :func:`assign`.
    n_clusters : int
        Number of clusters ``k``; clusters that receive no point get a zero count and a
        zero sum.
    sample_weight : array-like of shape (n_samples,), default=None
        Per-row weight; ``None`` means unit weights.

    Returns
    -------
    counts : ndarray of shape (n_clusters,), dtype float64
    sums : ndarray of shape (n_clusters, n_features), dtype float64
    """
    n_clusters = check_positive_integer("n_clusters", n_clusters)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != X.shape[0]:
        raise ValueError(
            "labels must have shape (n_samples,), got "
            f"{labels.shape} for X with {X.shape[0]} rows."
        )
    if labels.size and (labels.min() < 0 or labels.max() >= n_clusters):
        raise ValueError(
            f"labels must be in [0, n_clusters={n_clusters}), got "
            f"[{labels.min()}, {labels.max()}]."
        )
    weights = _prepare_weights(sample_weight, X.shape[0])

    counts = np.bincount(labels, weights=weights, minlength=n_clusters)
    counts = np.asarray(counts, dtype=np.float64)
    # Group the rows by cluster with a (k, n_samples) indicator matrix: this is O(nnz)
    # and works for dense and CSR batches with the same code.
    columns = np.arange(labels.shape[0])
    indicator = sp.csr_matrix(
        (weights, (labels, columns)), shape=(n_clusters, labels.shape[0])
    )
    if sp.issparse(X):
        sums = np.asarray((indicator @ X).toarray(), dtype=np.float64)
    else:
        sums = np.asarray(indicator @ _as_dense_float64(X), dtype=np.float64)
    return counts, sums


def _distances(X_chunk, centers64, centers_sq, x_squared_norms_chunk):
    """Squared distances between a chunk of rows and the centroids, in float64.

    Expanded form ``||x||**2 - 2 x.c + ||c||**2`` (D-028): it avoids materialising the
    ``(chunk, k, d)`` difference tensor.
    """
    if sp.issparse(X_chunk):
        dots = np.asarray(X_chunk @ centers64.T, dtype=np.float64)
    else:
        dots = _as_dense_float64(X_chunk) @ centers64.T
    return x_squared_norms_chunk[:, None] + centers_sq[None, :] - 2.0 * dots


def _prepare_centers(centers, X, x_squared_norms):
    """Validate the centroids; return them in float64, with their squared norms."""
    centers = np.asarray(centers)
    if centers.ndim != 2:
        raise ValueError(f"centers must be 2-dimensional, got shape {centers.shape}.")
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            f"centers has {centers.shape[1]} features but X has {X.shape[1]}."
        )
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must be finite.")
    centers64 = np.asarray(centers, dtype=np.float64)
    centers_sq = np.einsum("ij,ij->i", centers64, centers64)
    if x_squared_norms is None:
        x_squared_norms = row_squared_norms(X)
    else:
        x_squared_norms = np.asarray(x_squared_norms, dtype=np.float64).ravel()
        if x_squared_norms.shape[0] != X.shape[0]:
            raise ValueError(
                "x_squared_norms must hold one value per row of X, got "
                f"{x_squared_norms.shape[0]} for {X.shape[0]} rows."
            )
    return centers64, centers_sq, x_squared_norms


def _prepare_weights(sample_weight, n_samples):
    """Return the weights as a float64 array of shape ``(n_samples,)``."""
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (n_samples,):
        raise ValueError(
            f"sample_weight must have shape ({n_samples},), got {weights.shape}."
        )
    return weights


def _as_dense_float64(X):
    """Return ``X`` as a float64 ndarray (no copy if it already is one)."""
    return np.asarray(X, dtype=np.float64)


def _chunks(n_samples, chunk_size):
    """Yield ``(start, stop)`` row bounds covering ``[0, n_samples)``."""
    for start in range(0, n_samples, chunk_size):
        yield start, min(start + chunk_size, n_samples)


def distances(X, centers, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """Squared distances between every row of ``X`` and every centroid.

    Used by ``EMeans.transform``; the chunked loop keeps the (chunk, k) buffer
    bounded and is the same code path as :func:`assign`.

    Returns
    -------
    ndarray of shape (n_samples, n_clusters), dtype float64
    """
    centers64, centers_sq, x_squared_norms = _prepare_centers(centers, X, None)
    chunk_size = check_positive_integer("chunk_size", chunk_size)
    out = np.empty((X.shape[0], centers64.shape[0]), dtype=np.float64)
    for start, stop in _chunks(X.shape[0], chunk_size):
        out[start:stop] = _distances(
            X[start:stop], centers64, centers_sq, x_squared_norms[start:stop]
        )
    return out
