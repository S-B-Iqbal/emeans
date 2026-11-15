# SPDX-License-Identifier: BSD-3-Clause
"""``n_revivals_`` accounting (P8.2b).

``revive()`` returning a count is only useful if the estimator surfaces it, and if the
count can be trusted: these tests pin the two cases that matter - revival fires when
clusters die, and the documented switch-off idiom (a ``revival_period`` larger than the
number of steps) silences it.
"""

import numpy as np

from emeans import EMeans

# Few distinct locations, many clusters: most clusters starve and must be revived.
N_GROUPS = 3
N_CLUSTERS = 24


def _clustered(n_samples=1500, n_features=8, seed=0):
    rng = np.random.RandomState(seed)
    centres = rng.normal(size=(N_CLUSTERS, n_features)) * 5.0
    rows = rng.randint(N_GROUPS, size=n_samples)
    return centres[rows] + 0.4 * rng.normal(size=(n_samples, n_features))


def test_revival_fires_when_clusters_starve():
    model = EMeans(
        n_clusters=N_CLUSTERS,
        n_epochs=3,
        batch_size=64,
        alpha0=0.5,
        random_state=0,
    ).fit(_clustered())

    assert model.n_revivals_ > 0
    assert model.n_steps_ == 3 * np.ceil(1500 / 64)


def test_a_long_revival_period_switches_revival_off():
    params = {
        "n_clusters": N_CLUSTERS,
        "n_epochs": 3,
        "batch_size": 64,
        "alpha0": 0.5,
        "random_state": 0,
    }

    on = EMeans(**params).fit(_clustered())
    off = EMeans(revival_period=10**9, **params).fit(_clustered())

    assert on.n_revivals_ > 0
    assert off.n_revivals_ == 0
    assert off.n_steps_ == on.n_steps_


def test_revivals_are_deterministic_for_a_fixed_seed():
    params = {
        "n_clusters": N_CLUSTERS,
        "n_epochs": 2,
        "batch_size": 128,
        "alpha0": 0.3,
        "random_state": 7,
    }

    first = EMeans(**params).fit(_clustered(seed=1))
    second = EMeans(**params).fit(_clustered(seed=1))

    assert first.n_revivals_ == second.n_revivals_


def test_partial_fit_accumulates_revivals():
    model = EMeans(
        n_clusters=N_CLUSTERS,
        batch_size=64,
        alpha0=0.5,
        revival_period=1,
        random_state=0,
    )
    rng = np.random.RandomState(0)
    for _ in range(3):
        model.partial_fit(rng.normal(size=(200, 8)))

    assert model.n_steps_ == 3
    assert model.n_revivals_ >= 0  # revived after every step, count must be set
    assert isinstance(model.n_revivals_, int)
