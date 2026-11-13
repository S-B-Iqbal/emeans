# SPDX-License-Identifier: BSD-3-Clause
"""sample_weight behaviour (Phase 2, task P2.3, decisions D-024/D-041)."""

import numpy as np

from emeans import EMeans


def blobs(seed=0, n_per_cluster=20, n_features=4):
    rng = np.random.RandomState(seed)
    centers = np.zeros((3, n_features))
    centers[:, 0] = [-3.0, 0.0, 3.0]
    parts = [
        centers[j][None, :] + 0.3 * rng.normal(size=(n_per_cluster, n_features))
        for j in range(3)
    ]
    return np.vstack(parts)


START = np.array([[-3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])


def exact_regime(**kwargs):
    """alpha0 = 1 makes one partial_fit step exactly a weighted Lloyd step."""
    params = {
        "n_clusters": 3,
        "alpha0": 1.0,
        "init": START,
        "revival_period": 1000,
        "random_state": 0,
    }
    params.update(kwargs)
    return EMeans(**params)


def centres_with(X, weights=None):
    return exact_regime().partial_fit(X, sample_weight=weights).cluster_centers_


def test_integer_weights_equal_repeated_rows():
    X = blobs()
    weights = np.ones(X.shape[0], dtype=np.int64)
    weights[::3] = 2
    weighted = centres_with(X, weights)
    repeated = centres_with(np.repeat(X, weights, axis=0))
    np.testing.assert_allclose(weighted, repeated, rtol=1e-10)


def test_weight_scale_invariance():
    X = blobs()
    weights = np.linspace(0.5, 2.0, X.shape[0])
    np.testing.assert_allclose(
        centres_with(X, weights * 7.0), centres_with(X, weights), rtol=1e-12
    )


def test_zero_weight_rows_are_ignored():
    X = blobs()
    weights = np.ones(X.shape[0])
    weights[:5] = 0.0
    np.testing.assert_allclose(
        centres_with(X, weights), centres_with(X[5:]), rtol=1e-12
    )


def test_fit_with_weights_matches_repeated_rows_statistically():
    X = blobs()
    weights = np.random.RandomState(3).randint(1, 3, size=X.shape[0])
    repeated = np.repeat(X, weights, axis=0)
    relative = []
    for seed in range(3):
        params = {
            "n_clusters": 3,
            "n_epochs": 2,
            "batch_size": 32,
            "alpha0": 0.5,
            "random_state": seed,
        }
        left = EMeans(**params).fit(X, sample_weight=weights).inertia_
        right = EMeans(**params).fit(repeated).inertia_
        relative.append(abs(left - right) / right)
    assert np.mean(relative) < 0.05
