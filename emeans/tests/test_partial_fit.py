# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :meth:`emeans.EMeans.partial_fit` (Phase 1, task P1.8)."""

import numpy as np
import pytest

from emeans import EMeans
from emeans._scaling_laws import auto_alpha0


def blobs(seed=0, n_per_cluster=20, n_features=4):
    rng = np.random.RandomState(seed)
    centers = np.zeros((2, n_features))
    centers[:, 0] = [-3.0, 3.0]
    parts = [
        centers[j][None, :] + 0.3 * rng.normal(size=(n_per_cluster, n_features))
        for j in range(2)
    ]
    return np.vstack(parts)


def test_constant_step_is_used_verbatim_and_never_annealed():
    X = blobs()
    estimator = EMeans(n_clusters=2, alpha0=0.5, random_state=0)
    estimator.partial_fit(X)
    assert estimator.alpha0_ == 0.5
    assert estimator._partial_fit_step == 0.5
    assert estimator.n_steps_ == 1
    estimator.partial_fit(X)
    assert estimator.n_steps_ == 2
    assert estimator._partial_fit_step == 0.5


def test_auto_step_is_frozen_on_the_first_call():
    X = blobs()
    estimator = EMeans(n_clusters=2, alpha0="auto", random_state=0)
    estimator.partial_fit(X)
    expected = 0.1 * auto_alpha0(X.shape[0], 2, 1, X.shape[0])
    assert estimator._partial_fit_step == pytest.approx(expected, rel=1e-12)
    assert estimator.batch_size_ == X.shape[0]


def test_first_call_needs_at_least_n_clusters_rows():
    with pytest.raises(ValueError, match="on the first partial_fit call"):
        EMeans(n_clusters=8, random_state=0).partial_fit(np.zeros((3, 2)))


def test_feature_count_is_checked_across_calls():
    X = blobs()
    estimator = EMeans(n_clusters=2, alpha0=0.5, random_state=0).partial_fit(X)
    with pytest.raises(ValueError):
        estimator.partial_fit(np.zeros((10, 3)))


def test_a_step_of_one_gives_the_exact_weighted_means():
    X = blobs()
    centers = np.array([[-3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    estimator = EMeans(
        n_clusters=2,
        alpha0=1.0,
        init=centers,
        revival_period=1000,
        random_state=0,
    )
    estimator.partial_fit(X)
    labels = estimator.predict(X)
    for cluster in range(2):
        rows = X[labels == cluster]
        np.testing.assert_allclose(
            estimator.cluster_centers_[cluster], rows.mean(axis=0), rtol=1e-12
        )


def test_labels_and_inertia_describe_the_current_batch():
    X = blobs()
    estimator = EMeans(n_clusters=2, alpha0=0.5, random_state=0).partial_fit(X)
    assert estimator.labels_.shape == (X.shape[0],)
    assert estimator.inertia_ > 0.0
    assert estimator.score(X) == pytest.approx(-estimator.inertia_, rel=1e-12)


def test_the_revival_counter_continues_across_calls():
    X = blobs()
    estimator = EMeans(n_clusters=2, alpha0=0.5, revival_period=2, random_state=0)
    for _ in range(3):
        estimator.partial_fit(X)
    assert estimator.n_steps_ == 3


def test_partial_fit_after_fit_continues_the_stream():
    X = blobs()
    estimator = EMeans(
        n_clusters=2, n_epochs=1, batch_size=16, alpha0=0.5, random_state=0
    ).fit(X)
    steps_after_fit = estimator.n_steps_
    assert steps_after_fit == 3  # ceil(40 / 16)
    assert estimator._partial_fit_step == 0.5
    estimator.partial_fit(X)
    assert estimator.n_steps_ == steps_after_fit + 1


def test_partial_fit_after_fit_with_auto_params():
    X = blobs()
    estimator = EMeans(n_clusters=2, n_epochs=1, batch_size=16, random_state=0)
    estimator.fit(X)
    expected = 0.1 * estimator.alpha0_
    assert estimator._partial_fit_step == expected
    estimator.partial_fit(X)
    assert estimator.n_steps_ == 4
