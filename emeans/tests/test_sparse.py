# SPDX-License-Identifier: BSD-3-Clause
"""Dense vs CSR equivalence (Phase 2, task P2.2)."""

import numpy as np
import pytest
import scipy.sparse as sp

from emeans import EMeans


def sparse_blobs(seed=0, n_per_cluster=30, n_features=6):
    """Blobs with roughly half of the entries zeroed, so CSR is not trivial."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((3, n_features))
    centers[:, 0] = [-3.0, 0.0, 3.0]
    parts = [
        centers[j][None, :] + 0.3 * rng.normal(size=(n_per_cluster, n_features))
        for j in range(3)
    ]
    X = np.vstack(parts)
    X[rng.uniform(size=X.shape) < 0.5] = 0.0
    return X


def quick(**kwargs):
    params = {
        "n_clusters": 3,
        "n_epochs": 2,
        "batch_size": 16,
        "alpha0": 0.5,
        "random_state": 0,
    }
    params.update(kwargs)
    return EMeans(**params)


def test_dense_and_csr_agree():
    X = sparse_blobs()
    dense = quick().fit(X)
    sparse = quick().fit(sp.csr_matrix(X))
    np.testing.assert_allclose(
        dense.cluster_centers_, sparse.cluster_centers_, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(dense.labels_, sparse.labels_)
    assert dense.inertia_ == pytest.approx(sparse.inertia_, rel=1e-10)
    assert dense.n_steps_ == sparse.n_steps_


def test_sparse_transform_returns_a_dense_ndarray():
    X = sparse_blobs()
    transformed = quick().fit(sp.csr_matrix(X)).transform(sp.csr_matrix(X))
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (X.shape[0], 3)
    np.testing.assert_allclose(
        transformed, quick().fit(X).transform(X), rtol=1e-10, atol=1e-12
    )


def test_sparse_predict_matches_dense_predict():
    X = sparse_blobs()
    estimator = quick().fit(X)
    np.testing.assert_array_equal(
        estimator.predict(sp.csr_matrix(X)), estimator.predict(X)
    )


def test_all_zero_rows_stay_finite():
    X = sp.csr_matrix(np.zeros((6, 3)))
    estimator = quick(n_clusters=2, n_epochs=1).fit(X)
    assert np.all(np.isfinite(estimator.cluster_centers_))
    assert np.all(np.isfinite(estimator.transform(X)))
