# SPDX-License-Identifier: BSD-3-Clause
"""Parameter validation and error messages (Phase 2, task P2.4)."""

import warnings

import numpy as np
import pytest

from emeans import EMeans


def blobs(seed=0, n_per_cluster=10, n_features=4):
    rng = np.random.RandomState(seed)
    centers = np.zeros((3, n_features))
    centers[:, 0] = [-3.0, 0.0, 3.0]
    parts = [
        centers[j][None, :] + 0.3 * rng.normal(size=(n_per_cluster, n_features))
        for j in range(3)
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


@pytest.mark.parametrize(
    "params",
    [
        {"n_clusters": 0},
        {"n_epochs": 0},
        {"batch_size": 0},
        {"alpha0": 0.0},
        {"alpha0": 1.5},
        {"alpha_floor_ratio": 0.0},
        {"alpha_floor_ratio": 1.0},
        {"init": "unknown"},
        {"revival_threshold": 0.0},
        {"revival_period": 0},
        {"finalization": "unknown"},
        {"slow_ema_speed": 0.0},
    ],
)
def test_invalid_parameters_raise(params):
    with pytest.raises(ValueError):
        quick(**params).fit(blobs())


def test_too_few_samples_message_matches_kmeans_style():
    with pytest.raises(ValueError, match=r"n_samples=2 should be >= n_clusters=3"):
        quick().fit(np.zeros((2, 4)))


def test_init_shape_messages():
    X = blobs()
    with pytest.raises(ValueError, match="number of clusters"):
        quick(init=np.zeros((4, 4))).fit(X)
    with pytest.raises(ValueError, match="number of features"):
        quick(init=np.zeros((3, 5))).fit(X)


def test_out_of_range_warning_only_for_auto_parameters():
    X = blobs()
    with pytest.warns(UserWarning, match="scaling laws"):
        EMeans(n_clusters=3, random_state=0).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quick().fit(X)


def test_fitted_state_is_consistent():
    # finalization=None keeps the returned centroids equal to S / N (D-009)
    estimator = quick(finalization=None).fit(blobs())
    np.testing.assert_allclose(
        estimator.cluster_centers_,
        estimator._sums / estimator._counts[:, None],
        rtol=1e-12,
    )
    assert np.all(estimator._counts > 0.0)
    assert estimator._sums.shape == (3, 4)
