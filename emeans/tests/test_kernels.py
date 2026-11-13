# SPDX-License-Identifier: BSD-3-Clause
"""Build-level checks for the compiled kernels (Phase 3, task P3.1)."""

import numpy as np

from emeans._kernels import _IMPORT_ERROR, HAS_KERNELS, has_openmp, max_threads


def test_compiled_kernels_are_available():
    # Every install path of this project builds the extension (wheel and editable).
    assert HAS_KERNELS is True, _IMPORT_ERROR


def test_probe_reports_a_boolean_and_a_thread_count():
    assert isinstance(has_openmp(), bool)
    assert max_threads() >= 1


def test_thread_count_matches_the_openmp_flag():
    if has_openmp():
        assert max_threads() >= 1
    else:
        assert max_threads() == 1


# --- assignment kernels (P3.2) ---------------------------------------------


def blobs(seed=0, n_samples=500, n_clusters=7, n_features=4, spread=0.2):
    rng = np.random.RandomState(seed)
    centers = rng.normal(size=(n_clusters, n_features))
    rows = rng.randint(0, n_clusters, n_samples)
    return centers[rows] + spread * rng.normal(size=(n_samples, n_features)), centers


def test_assign_dense_matches_the_numpy_oracle():
    from emeans._kernels._assign import assign_dense
    from emeans._numpy_backend import assign, row_squared_norms

    for seed in range(3):
        X, centers = blobs(seed=seed)
        X = np.ascontiguousarray(X)
        centers = np.ascontiguousarray(centers)
        labels = np.zeros(X.shape[0], dtype=np.int32)
        assign_dense(X, centers, row_squared_norms(X), labels, 2)
        np.testing.assert_array_equal(labels, assign(X, centers))


def test_assign_csr_matches_the_numpy_oracle():
    import scipy.sparse as sp

    from emeans._kernels._assign import assign_csr
    from emeans._numpy_backend import assign, row_squared_norms

    X, centers = blobs(seed=1)
    centers = np.ascontiguousarray(centers)
    sparse = sp.csr_matrix(X)
    labels = np.zeros(X.shape[0], dtype=np.int32)
    assign_csr(
        sparse.data,
        sparse.indices.astype(np.int32),
        sparse.indptr.astype(np.int32),
        centers,
        row_squared_norms(sparse),
        labels,
        2,
    )
    np.testing.assert_array_equal(labels, assign(X, centers))


def test_assign_handles_multiple_chunks():
    from emeans._kernels import _common
    from emeans._kernels._assign import assign_dense
    from emeans._numpy_backend import assign, row_squared_norms

    # more rows than CHUNK_SIZE, so the remainder branch is exercised
    X, centers = blobs(seed=2, n_samples=3 * _common.CHUNK_SIZE + 5)
    X = np.ascontiguousarray(X)
    centers = np.ascontiguousarray(centers)
    labels = np.zeros(X.shape[0], dtype=np.int32)
    assign_dense(X, centers, row_squared_norms(X), labels, 3)
    np.testing.assert_array_equal(labels, assign(X, centers))


def test_assign_float32_specialisation():
    from emeans._kernels._assign import assign_dense
    from emeans._numpy_backend import assign, row_squared_norms

    X, centers = blobs(seed=3)
    X = np.ascontiguousarray(X, dtype=np.float32)
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    labels = np.zeros(X.shape[0], dtype=np.int32)
    assign_dense(X, centers, row_squared_norms(X), labels, 1)
    np.testing.assert_array_equal(labels, assign(X, centers))


def test_assign_breaks_ties_to_the_lowest_index():
    from emeans._kernels._assign import assign_dense

    X = np.array([[1.0], [0.5]])
    centers = np.array([[0.0], [2.0]])
    labels = np.zeros(2, dtype=np.int32)
    assign_dense(
        np.ascontiguousarray(X),
        np.ascontiguousarray(centers),
        np.array([1.0, 0.25]),
        labels,
        1,
    )
    np.testing.assert_array_equal(labels, [0, 0])


# --- reduction kernels (P3.3) ----------------------------------------------


def reduction_problem(seed=0, n_samples=700, n_clusters=5, n_features=3):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, n_clusters, n_samples).astype(np.int32)
    weights = rng.uniform(0.5, 2.0, n_samples)
    X = np.ascontiguousarray(rng.normal(size=(n_samples, n_features)))
    return X, labels, weights


def test_reduce_dense_matches_the_numpy_oracle():
    from emeans._kernels._reduce import reduce_dense
    from emeans._numpy_backend import reduce_batch

    X, labels, weights = reduction_problem()
    counts, sums = reduce_batch(X, labels, 5, weights)
    got_counts = np.zeros(5)
    got_sums = np.zeros((5, 3))
    reduce_dense(X, labels, weights, got_counts, got_sums, 4)
    np.testing.assert_allclose(got_counts, counts, rtol=1e-12)
    np.testing.assert_allclose(got_sums, sums, rtol=1e-12)


def test_reduce_csr_matches_the_numpy_oracle():
    import scipy.sparse as sp

    from emeans._kernels._reduce import reduce_csr
    from emeans._numpy_backend import reduce_batch

    X, labels, weights = reduction_problem(seed=1)
    counts, sums = reduce_batch(X, labels, 5, weights)
    sparse = sp.csr_matrix(X)
    got_counts = np.zeros(5)
    got_sums = np.zeros((5, 3))
    reduce_csr(
        sparse.data,
        sparse.indices.astype(np.int32),
        sparse.indptr.astype(np.int32),
        labels,
        weights,
        got_counts,
        got_sums,
        4,
    )
    np.testing.assert_allclose(got_counts, counts, rtol=1e-12)
    np.testing.assert_allclose(got_sums, sums, rtol=1e-12)


def test_reduce_ignores_zero_weight_rows():
    from emeans._kernels._reduce import reduce_dense
    from emeans._numpy_backend import reduce_batch

    X, labels, weights = reduction_problem(seed=2)
    weights[:50] = 0.0
    expected_counts, expected_sums = reduce_batch(X, labels, 5, weights)
    counts = np.zeros(5)
    sums = np.zeros((5, 3))
    reduce_dense(X, labels, weights, counts, sums, 2)
    np.testing.assert_allclose(counts, expected_counts, rtol=1e-12)
    np.testing.assert_allclose(sums, expected_sums, rtol=1e-12)
