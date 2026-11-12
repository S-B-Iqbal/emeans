# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :mod:`emeans._numpy_backend` (Phase 1, task P1.3).

The reference used here is written as literally as possible (difference form, Python
loops for the reduction) so that it does not share the expanded-form arithmetic of the
module under test.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from emeans._numpy_backend import (
    CHUNK_SIZE,
    assign,
    labels_inertia,
    reduce_batch,
    row_squared_norms,
)


def make_problem(seed=0, n_samples=60, n_clusters=4, n_features=3):
    """Well-separated blobs, so float32 and float64 agree on the assignment."""
    rng = np.random.RandomState(seed)
    centers = rng.normal(size=(n_clusters, n_features))
    rows = rng.randint(0, n_clusters, size=n_samples)
    X = centers[rows] + 0.1 * rng.normal(size=(n_samples, n_features))
    return X, centers


def brute_force_labels(X, centers):
    """Literal ``argmin_j ||x - c_j||**2`` with the difference form."""
    X64 = np.asarray(X, dtype=np.float64)
    C64 = np.asarray(centers, dtype=np.float64)
    diff = X64[:, None, :] - C64[None, :, :]
    return np.argmin((diff**2).sum(axis=-1), axis=1)


def brute_force_inertia(X, centers, weights=None):
    X64 = np.asarray(X, dtype=np.float64)
    C64 = np.asarray(centers, dtype=np.float64)
    diff = X64[:, None, :] - C64[None, :, :]
    d2 = (diff**2).sum(axis=-1)
    nearest = d2.min(axis=1)
    if weights is None:
        return float(nearest.sum())
    return float(np.dot(np.asarray(weights, dtype=np.float64), nearest))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_assign_matches_brute_force(dtype):
    for seed in range(3):
        X, centers = make_problem(seed=seed)
        labels = assign(X.astype(dtype), centers.astype(dtype))
        np.testing.assert_array_equal(labels, brute_force_labels(X, centers))
        assert labels.dtype == np.int32


def test_assign_breaks_ties_to_the_lowest_index():
    # x = 1.0 is exactly equidistant from the two centroids -> the lowest index wins.
    np.testing.assert_array_equal(
        assign(np.array([[1.0]]), np.array([[0.0], [2.0]])), [0]
    )
    # x = 0.5 sits exactly between 0.0 and 1.0 -> the lowest index wins again.
    np.testing.assert_array_equal(
        assign(np.array([[0.5]]), np.array([[0.0], [1.0]])), [0]
    )
    # With a third centroid at 1.0 the closest one is index 2, not a tie.
    centers = np.array([[0.0], [2.0], [1.0]])
    np.testing.assert_array_equal(assign(np.array([[1.0]]), centers), [2])


def test_assign_is_chunk_size_invariant():
    X, centers = make_problem(seed=7, n_samples=37)
    reference = assign(X, centers, chunk_size=CHUNK_SIZE)
    for chunk_size in (1, 2, 5, 36, 37, 100):
        np.testing.assert_array_equal(
            assign(X, centers, chunk_size=chunk_size), reference
        )


def test_assign_accepts_precomputed_row_norms():
    X, centers = make_problem(seed=11)
    norms = row_squared_norms(X)
    np.testing.assert_array_equal(
        assign(X, centers, norms), brute_force_labels(X, centers)
    )


def test_assign_works_on_csr_input():
    X, centers = make_problem(seed=3, n_samples=40, n_clusters=3, n_features=5)
    X_sparse = sp.csr_matrix(X)
    assert sp.issparse(X_sparse)
    np.testing.assert_array_equal(
        assign(X_sparse, centers), brute_force_labels(X, centers)
    )
    # The sparse path must also accept the precomputed norms of the CSR matrix itself.
    norms = row_squared_norms(X_sparse)
    np.testing.assert_array_equal(
        assign(X_sparse, centers, norms), assign(X_sparse, centers)
    )


def test_row_squared_norms_dense_and_csr_agree():
    X, _ = make_problem(seed=5, n_features=4)
    expected = np.einsum("ij,ij->i", X, X)
    np.testing.assert_allclose(row_squared_norms(X), expected, rtol=1e-15)
    np.testing.assert_allclose(
        row_squared_norms(sp.csr_matrix(X)), expected, rtol=1e-15
    )
    assert row_squared_norms(X).dtype == np.float64


def test_assign_validates_its_inputs():
    X, centers = make_problem(seed=0)
    with pytest.raises(ValueError):
        assign(X, np.array([0.0, 1.0, 2.0]))  # centroids must be 2-dimensional
    with pytest.raises(ValueError):
        assign(X, centers[:, :-1])  # wrong number of features
    with pytest.raises(ValueError):
        assign(X, np.full((4, 3), np.nan))
    with pytest.raises(ValueError):
        assign(X, centers, x_squared_norms=np.ones(3))
    with pytest.raises(ValueError):
        assign(X, centers, chunk_size=0)
    with pytest.raises(TypeError):
        assign(X, centers, chunk_size=1.5)
    assert CHUNK_SIZE == 256  # mirrors sklearn's chunked kernels


def test_reduce_batch_matches_a_loop_reference():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(17, 3))
    labels = rng.randint(0, 4, size=17)
    weights = rng.uniform(0.5, 2.0, size=17)

    counts, sums = reduce_batch(X, labels, 4, weights)
    assert counts.shape == (4,) and sums.shape == (4, 3)
    assert counts.dtype == np.float64 and sums.dtype == np.float64

    expected_counts = np.zeros(4)
    expected_sums = np.zeros((4, 3))
    for row, cluster in enumerate(labels):
        expected_counts[cluster] += weights[row]
        expected_sums[cluster] += weights[row] * X[row]
    np.testing.assert_allclose(counts, expected_counts, rtol=1e-12)
    np.testing.assert_allclose(sums, expected_sums, rtol=1e-12)


def test_reduce_batch_leaves_unused_clusters_empty():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = np.zeros(3, dtype=np.int32)
    counts, sums = reduce_batch(X, labels, 3)
    np.testing.assert_array_equal(counts, [3.0, 0.0, 0.0])
    np.testing.assert_array_equal(sums, [[9.0, 12.0], [0.0, 0.0], [0.0, 0.0]])


def test_reduce_batch_ignores_zero_weight_rows():
    X = np.array([[1.0, 1.0], [5.0, 5.0]])
    labels = np.zeros(2, dtype=np.int32)
    counts, sums = reduce_batch(X, labels, 1, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(counts, [1.0])
    np.testing.assert_array_equal(sums, [[1.0, 1.0]])


def test_reduce_batch_csr_matches_dense():
    X, centers = make_problem(seed=4, n_samples=30, n_clusters=3, n_features=4)
    labels = assign(X, centers)
    weights = np.linspace(0.5, 1.5, X.shape[0])
    dense_counts, dense_sums = reduce_batch(X, labels, 3, weights)
    sparse_counts, sparse_sums = reduce_batch(sp.csr_matrix(X), labels, 3, weights)
    np.testing.assert_allclose(sparse_counts, dense_counts, rtol=1e-12)
    np.testing.assert_allclose(sparse_sums, dense_sums, rtol=1e-12)


def test_reduce_batch_validates_its_inputs():
    X = np.zeros((4, 3))
    labels = np.zeros(4, dtype=np.int32)
    with pytest.raises(ValueError):
        reduce_batch(X, labels[:3], 2)  # one label per row is required
    with pytest.raises(ValueError):
        reduce_batch(X, np.array([0, 1, 2, 7], dtype=np.int32), 2)  # out of range
    with pytest.raises(ValueError):
        reduce_batch(X, labels, 2, np.ones(3))
    with pytest.raises(ValueError):
        reduce_batch(X, labels, 0)


def test_labels_inertia_matches_brute_force():
    X, centers = make_problem(seed=9)
    weights = np.linspace(0.5, 2.0, X.shape[0])
    labels, inertia = labels_inertia(X, centers, weights)
    np.testing.assert_array_equal(labels, brute_force_labels(X, centers))
    assert inertia == pytest.approx(
        brute_force_inertia(X, centers, weights), rel=1e-12, abs=1e-15
    )


def test_labels_inertia_unit_weights_and_chunk_invariance():
    X, centers = make_problem(seed=1, n_samples=23)
    labels, inertia = labels_inertia(X, centers)
    chunked_labels, chunked_inertia = labels_inertia(
        X, centers, np.ones(X.shape[0]), chunk_size=3
    )
    np.testing.assert_array_equal(labels, chunked_labels)
    assert inertia == pytest.approx(chunked_inertia, rel=1e-15)
    assert inertia == pytest.approx(brute_force_inertia(X, centers), rel=1e-12)


def test_labels_inertia_accepts_float32_input():
    X, centers = make_problem(seed=13)
    labels, inertia = labels_inertia(X.astype(np.float32), centers.astype(np.float32))
    np.testing.assert_array_equal(labels, brute_force_labels(X, centers))
    assert inertia == pytest.approx(brute_force_inertia(X, centers), rel=1e-5)
