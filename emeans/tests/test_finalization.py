# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the finalization strategies (Phase 1, task P1.6)."""

import numpy as np

from emeans import EMeans
from emeans._numpy_backend import labels_inertia


def blobs(seed=0, n_per_cluster=40, n_clusters=3, n_features=4):
    rng = np.random.RandomState(seed)
    centers = np.zeros((n_clusters, n_features))
    centers[:, 0] = np.linspace(-3.0, 3.0, n_clusters)
    parts = [
        centers[j][None, :] + 0.3 * rng.normal(size=(n_per_cluster, n_features))
        for j in range(n_clusters)
    ]
    return np.vstack(parts)


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


def train_inertia(estimator, X):
    _, inertia = labels_inertia(X, estimator.cluster_centers_)
    return inertia


def test_last_epoch_mean_does_not_increase_train_inertia():
    X = blobs()
    ema = quick(finalization=None).fit(X)
    closing = quick(finalization="last_epoch_mean").fit(X)
    assert train_inertia(closing, X) <= train_inertia(ema, X) + 1e-9


def test_none_finalization_returns_the_ema_centroids():
    X = blobs()
    ema = quick(finalization=None).fit(X)
    np.testing.assert_allclose(
        ema.cluster_centers_, ema._sums / ema._counts[:, None], rtol=1e-12
    )


def test_slow_ema_is_not_used_for_assignment():
    X = blobs()
    slow = quick(finalization="slow_ema", slow_ema_speed=0.25).fit(X)
    fast = quick(finalization="slow_ema", slow_ema_speed=4.0).fit(X)
    np.testing.assert_array_equal(slow.labels_, fast.labels_)
    np.testing.assert_allclose(slow._counts, fast._counts, rtol=1e-12)
    assert not np.allclose(slow.cluster_centers_, fast.cluster_centers_, rtol=1e-6)


def test_slow_ema_centroids_are_the_slow_state():
    X = blobs()
    estimator = quick(finalization="slow_ema", slow_ema_speed=0.25).fit(X)
    np.testing.assert_allclose(
        estimator.cluster_centers_,
        estimator._slow_sums / estimator._slow_counts[:, None],
        rtol=1e-12,
    )


def test_last_epoch_mean_falls_back_to_the_ema_centroid():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    weights = np.ones(X.shape[0])
    estimator = quick(n_clusters=3)
    estimator._n_threads = 1  # _last_epoch_mean is called after fit in practice
    # cluster 2 receives no point during the last epoch
    estimator._last_epoch_labels = np.array([0, 0, 1], dtype=np.int32)
    fallback = np.array([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]])
    means = estimator._last_epoch_mean(X, weights, fallback)
    np.testing.assert_allclose(means[0], [0.5, 0.0])
    np.testing.assert_allclose(means[1], [2.0, 0.0])
    np.testing.assert_allclose(means[2], fallback[2])


def test_last_epoch_mean_differs_from_the_ema_centroids():
    X = blobs()
    ema = quick(finalization=None).fit(X)
    closing = quick(finalization="last_epoch_mean").fit(X)
    assert not np.allclose(closing.cluster_centers_, ema.cluster_centers_)
