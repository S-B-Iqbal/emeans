# SPDX-License-Identifier: BSD-3-Clause
"""Datasets for the benchmark suite (Phase 4, task P4.1).

Synthetic generators keep the suite offline and reproducible; the held-out split follows
rule 09: up to 20 000 vectors are removed from the training data, deduplicated and split
50/50 into validation and test.
"""

import numpy as np

MAX_HELDOUT = 20_000


def make_blobs(n_samples, n_features, n_clusters, seed=0, spread=1.0, noise=0.3):
    """Isotropic Gaussian blobs: the easy, embedding-like case."""
    rng = np.random.RandomState(seed)
    centers = rng.normal(size=(n_clusters, n_features)) * spread
    rows = rng.randint(0, n_clusters, n_samples)
    return centers[rows] + noise * rng.normal(size=(n_samples, n_features))


def make_low_dimension(n_samples, seed=0):
    """Two-dimensional data: not the regime the laws were fitted for."""
    return make_blobs(n_samples, n_features=2, n_clusters=8, seed=seed)


def make_high_n_over_k(n_samples, n_clusters, seed=0):
    """n / k = 1000, the Table 10 regime."""
    return make_blobs(n_samples, n_features=16, n_clusters=n_clusters, seed=seed)


def held_out_split(X, seed=0, max_heldout=MAX_HELDOUT):
    """Return (train, validation, test) with a deduplicated held-out set."""
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    n_hold = min(max_heldout, n_samples // 2)
    perm = rng.permutation(n_samples)
    held = np.unique(X[perm[:n_hold]], axis=0)
    n_val = held.shape[0] // 2
    return X[perm[n_hold:]], held[:n_val], held[n_val:]
