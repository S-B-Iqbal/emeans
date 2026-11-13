# SPDX-License-Identifier: BSD-3-Clause
"""Measurement helpers for the benchmark suite (Phase 4, tasks P4.1/P4.2)."""

import time

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

from emeans import EMeans


def mse(X, centers):
    """Mean squared distance of the rows of ``X`` to their nearest centre."""
    _, distances = pairwise_distances_argmin_min(X, centers)
    return float(np.mean(distances**2))


def relative_mse(value, reference):
    """``(MSE - reference) / reference``, the metric of Sec. 4.1."""
    return (value - reference) / reference


def lloyd_at_25(X, n_clusters, seed, max_iter=25):
    """The reference of rule 09: Lloyd, random init, 25 iterations, no tolerance."""
    model = KMeans(
        n_clusters=n_clusters,
        init="random",
        algorithm="lloyd",
        n_init=1,
        max_iter=max_iter,
        tol=0.0,
        random_state=seed,
    )
    start = time.perf_counter()
    model.fit(X)
    return model.cluster_centers_, time.perf_counter() - start


def emeans_at(X, n_clusters, epochs, seed, **kwargs):
    """EMeans with automatic hyper-parameters: never tuned (rule 09)."""
    model = EMeans(
        n_clusters=n_clusters,
        n_epochs=epochs,
        random_state=seed,
        **kwargs,
    )
    start = time.perf_counter()
    model.fit(X)
    return model.cluster_centers_, time.perf_counter() - start, model


def minibatch_sweep(
    X_train,
    X_val,
    n_clusters,
    seed,
    max_iter=5,
    batch_sizes=(256, 1024, 4096),
    ratios=(0.0, 0.01, 0.1),
):
    """Sweep MiniBatchKMeans on validation, as in the paper Sculley arm (App. C).

    Early stopping is switched off (``max_no_improvement=None``, ``tol=0``) so every
    run completes its budget; the best configuration on the validation set is
    returned.  ``max_iter`` counts passes over the data, so it matches the epoch
    budget of the other methods.
    """
    best = None
    for batch_size in batch_sizes:
        if batch_size > X_train.shape[0]:
            continue
        for ratio in ratios:
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                init="random",
                n_init=1,
                max_iter=max_iter,
                batch_size=batch_size,
                reassignment_ratio=ratio,
                max_no_improvement=None,
                tol=0.0,
                random_state=seed,
            )
            model.fit(X_train)
            score = mse(X_val, model.cluster_centers_)
            if best is None or score < best[0]:
                best = (score, batch_size, ratio, model.cluster_centers_)
    return best
