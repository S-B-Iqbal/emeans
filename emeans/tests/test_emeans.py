# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :class:`emeans.EMeans` (Phase 1, tasks P1.5 and P1.7)."""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError

from emeans import EMeans
from emeans._numpy_backend import labels_inertia


def blobs(seed=0, n_per_cluster=40, n_clusters=3, n_features=4):
    """Three well-separated Gaussian blobs."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((n_clusters, n_features))
    centers[:, 0] = np.linspace(-3.0, 3.0, n_clusters)
    parts = [
        centers[j][None, :] + 0.3 * rng.normal(size=(n_per_cluster, n_features))
        for j in range(n_clusters)
    ]
    return np.vstack(parts)


def quick(**kwargs):
    """A fast, fully explicit configuration (no auto laws, no warning)."""
    params = {
        "n_clusters": 3,
        "n_epochs": 2,
        "batch_size": 16,
        "alpha0": 0.5,
        "random_state": 0,
    }
    params.update(kwargs)
    return EMeans(**params)


def test_fit_recovers_the_blobs():
    X = blobs()
    estimator = quick().fit(X)
    assert estimator.cluster_centers_.shape == (3, 4)
    assert estimator.cluster_centers_.dtype == np.float64
    assert estimator.labels_.shape == (X.shape[0],)
    assert estimator.labels_.dtype == np.int32
    assert estimator.inertia_ > 0.0
    found = np.sort(estimator.cluster_centers_[:, 0])
    np.testing.assert_allclose(found, [-3.0, 0.0, 3.0], atol=0.5)


def test_step_and_epoch_counters():
    X = blobs()
    estimator = quick(n_epochs=3, batch_size=32).fit(X)
    # n = 120, B = 32 -> 4 batches per epoch, tail kept (D-004)
    assert estimator.n_steps_ == 3 * 4
    assert estimator.n_iter_ == 3


def test_explicit_hyperparameters_are_used_as_given():
    X = blobs()
    estimator = quick(batch_size=7, alpha0=0.25).fit(X)
    assert estimator.batch_size_ == 7
    assert estimator.alpha0_ == 0.25


def test_auto_hyperparameters_are_resolved_and_warn_out_of_range():
    X = blobs()
    with pytest.warns(UserWarning, match="scaling laws were fitted"):
        estimator = EMeans(n_clusters=3, n_epochs=2, random_state=0).fit(X)
    assert estimator.batch_size_ == 4  # Eq. 7 on n=120, k=3, T=2
    assert 0.0 < estimator.alpha0_ <= 1.0


def test_explicit_parameters_do_not_warn():
    X = blobs()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quick().fit(X)


def test_fit_is_deterministic_for_a_fixed_seed():
    X = blobs()
    first = quick().fit(X)
    second = quick().fit(X)
    np.testing.assert_array_equal(first.cluster_centers_, second.cluster_centers_)
    np.testing.assert_array_equal(first.labels_, second.labels_)
    assert first.n_steps_ == second.n_steps_


def test_different_seeds_give_different_results():
    X = blobs()
    first = quick(random_state=0).fit(X)
    second = quick(random_state=1).fit(X)
    assert not np.allclose(first.cluster_centers_, second.cluster_centers_)


def test_unit_weights_match_no_weights():
    X = blobs()
    unweighted = quick().fit(X)
    weighted = quick().fit(X, sample_weight=np.ones(X.shape[0]))
    np.testing.assert_allclose(
        weighted.cluster_centers_, unweighted.cluster_centers_, rtol=1e-14
    )


def test_sparse_input_runs():
    X = sp.csr_matrix(blobs())
    estimator = quick().fit(X)
    assert estimator.cluster_centers_.shape == (3, 4)
    assert estimator.labels_.shape == (X.shape[0],)


def test_too_few_samples_raise():
    X = np.zeros((2, 3))
    with pytest.raises(ValueError, match="n_samples=2 should be >= n_clusters=3"):
        quick().fit(X)


def test_init_array_is_validated():
    X = blobs()
    centers = np.array([[-3.0, 0, 0, 0], [0.0, 0, 0, 0], [3.0, 0, 0, 0]])
    estimator = quick(init=centers, n_epochs=1, batch_size=120, alpha0=1.0).fit(X)
    np.testing.assert_allclose(estimator.cluster_centers_, centers, atol=1.0)
    with pytest.raises(ValueError, match="match the number of clusters"):
        quick(init=np.zeros((4, 4))).fit(X)
    with pytest.raises(ValueError, match="number of features"):
        quick(init=np.zeros((3, 5))).fit(X)


def test_kmeans_plus_plus_init_runs():
    X = blobs()
    estimator = quick(init="k-means++").fit(X)
    assert estimator.cluster_centers_.shape == (3, 4)


def test_verbose_prints_one_line_per_epoch(capsys):
    X = blobs()
    quick(n_epochs=2, verbose=1).fit(X)
    out = capsys.readouterr().out
    assert out.count("Epoch ") == 2
    assert "steps 16/16" in out  # 120 / 16 = 7.5 -> 8 batches per epoch


def test_labels_match_predict():
    X = blobs()
    estimator = quick().fit(X)
    np.testing.assert_array_equal(estimator.labels_, estimator.predict(X))


def test_transform_returns_euclidean_distances():
    X = blobs()
    estimator = quick().fit(X)
    transformed = estimator.transform(X)
    assert transformed.shape == (X.shape[0], 3)
    assert transformed.dtype == X.dtype
    differences = X[:, None, :] - estimator.cluster_centers_[None, :, :]
    expected = np.sqrt((differences**2).sum(axis=-1))
    np.testing.assert_allclose(transformed, expected, rtol=1e-9, atol=1e-9)


def test_float32_is_preserved():
    X = blobs().astype(np.float32)
    estimator = quick().fit(X)
    assert estimator.cluster_centers_.dtype == np.float32
    assert estimator.transform(X).dtype == np.float32
    assert estimator.predict(X).dtype == np.int32


def test_score_is_minus_the_inertia():
    X = blobs()
    estimator = quick().fit(X)
    assert estimator.score(X) == pytest.approx(-estimator.inertia_, rel=1e-12)
    weights = np.linspace(0.5, 1.5, X.shape[0])
    expected = labels_inertia(X, estimator.cluster_centers_, weights)[1]
    assert estimator.score(X, sample_weight=weights) == pytest.approx(
        -expected, rel=1e-12
    )


def test_fit_predict_and_fit_transform():
    X = blobs()
    labels = quick().fit_predict(X)
    assert labels.shape == (X.shape[0],)
    transformed = quick().fit_transform(X)
    assert transformed.shape == (X.shape[0], 3)


def test_feature_names_come_from_the_mixin():
    estimator = quick().fit(blobs())
    assert list(estimator.get_feature_names_out()) == [
        "emeans0",
        "emeans1",
        "emeans2",
    ]


def test_predict_and_transform_require_a_fitted_model():
    with pytest.raises(NotFittedError):
        quick().predict(blobs())
    with pytest.raises(NotFittedError):
        quick().transform(blobs())


def test_predict_rejects_a_different_feature_count():
    X = blobs()
    estimator = quick().fit(X)
    with pytest.raises(ValueError):
        estimator.predict(np.zeros((5, 7)))


def test_float32_and_float64_agree():
    X64 = blobs()
    X32 = X64.astype(np.float32)
    first = quick().fit(X64)
    second = quick().fit(X32)
    assert second.inertia_ == pytest.approx(first.inertia_, rel=1e-3)
    np.testing.assert_allclose(
        first.cluster_centers_, second.cluster_centers_, rtol=1e-3, atol=1e-3
    )


def test_float32_keeps_the_labels_stable():
    X64 = blobs()
    first = quick().fit(X64)
    second = quick().fit(X64.astype(np.float32))
    np.testing.assert_array_equal(first.labels_, second.labels_)


def test_array_like_input_is_accepted():
    X = blobs()
    as_list = X.tolist()
    estimator = quick().fit(as_list)
    assert estimator.cluster_centers_.shape == (3, 4)
    assert estimator.predict(as_list).shape == (X.shape[0],)
    assert estimator.score(as_list) == pytest.approx(-estimator.inertia_, rel=1e-12)


def test_fit_transform_accepts_array_like_input():
    X = blobs()
    transformed = quick().fit_transform(X.tolist())
    assert transformed.shape == (X.shape[0], 3)
    assert transformed.dtype == np.float64
