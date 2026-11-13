# SPDX-License-Identifier: BSD-3-Clause
"""NumPy vs Cython backend equality (Phase 3, task P3.5)."""

import numpy as np
import pytest
import scipy.sparse as sp

from emeans import EMeans, _backend


@pytest.fixture(autouse=True)
def restore_backend():
    """Leave the previously active backend in place after each test."""
    previous = _backend.get_backend()
    yield
    _backend.set_backend(previous)


def blobs(seed=0, n_samples=60, n_clusters=3, n_features=5):
    rng = np.random.RandomState(seed)
    centers = rng.normal(size=(n_clusters, n_features))
    rows = rng.randint(0, n_clusters, n_samples)
    return centers[rows] + 0.2 * rng.normal(size=(n_samples, n_features)), centers


def test_the_compiled_backend_is_available():
    assert _backend.get_backend() == "cython"


def test_set_backend_validates_its_argument():
    with pytest.raises(ValueError):
        _backend.set_backend("blas")


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_assignment_matches_between_backends(as_sparse, dtype):
    X, centers = blobs()
    X = X.astype(dtype)
    centers = centers.astype(dtype)
    data = sp.csr_matrix(X) if as_sparse else X
    _backend.set_backend("numpy")
    expected = _backend.assign(data, centers)
    _backend.set_backend("cython")
    got = _backend.assign(data, centers)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.int32


@pytest.mark.parametrize("as_sparse", [False, True])
def test_reduction_matches_between_backends(as_sparse):
    X, centers = blobs(n_samples=200)
    weights = np.linspace(0.5, 1.5, X.shape[0])
    labels = _backend.assign(X, centers).astype(np.int32)
    data = sp.csr_matrix(X) if as_sparse else X
    _backend.set_backend("numpy")
    counts_np, sums_np = _backend.reduce_batch(data, labels, 3, weights)
    _backend.set_backend("cython")
    counts_cy, sums_cy = _backend.reduce_batch(data, labels, 3, weights)
    np.testing.assert_allclose(counts_cy, counts_np, rtol=1e-12)
    np.testing.assert_allclose(sums_cy, sums_np, rtol=1e-12)


def test_estimator_end_to_end_matches_between_backends():
    X, _ = blobs(n_samples=300, n_clusters=4)
    params = {
        "n_clusters": 4,
        "n_epochs": 2,
        "batch_size": 32,
        "alpha0": 0.5,
        "random_state": 0,
    }
    _backend.set_backend("numpy")
    reference = EMeans(**params).fit(X)
    _backend.set_backend("cython")
    compiled = EMeans(**params).fit(X)
    np.testing.assert_array_equal(compiled.labels_, reference.labels_)
    assert compiled.n_steps_ == reference.n_steps_
    assert compiled.n_iter_ == reference.n_iter_
    np.testing.assert_allclose(
        compiled.cluster_centers_,
        reference.cluster_centers_,
        rtol=1e-10,
        atol=1e-12,
    )
    assert compiled.inertia_ == pytest.approx(reference.inertia_, rel=1e-10)


def test_estimator_streams_on_both_backends():
    X, _ = blobs(n_samples=120, n_clusters=3)
    outputs = []
    for name in ("numpy", "cython"):
        _backend.set_backend(name)
        model = EMeans(n_clusters=3, alpha0=0.5, random_state=0)
        model.partial_fit(X)
        outputs.append(model.cluster_centers_)
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-10, atol=1e-12)
