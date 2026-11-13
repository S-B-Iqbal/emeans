# SPDX-License-Identifier: BSD-3-Clause
"""Dead-cluster revival (Sec. 3.1, App. H.1, Eq. 8).

Called every ``revival_period`` updates. Clusters whose EMA count
``N_j`` fell below ``tau`` times the mean count are rebuilt by
splitting a donor cluster (D-012, D-013):

1. a donor is drawn without replacement, proportional to its count;
2. its centroid gets two complementary perturbations (Eq. 8): one
   child takes the dead slot, the other replaces the donor;
3. both counts become half of the donor count and both sums are
   rewritten as ``S = c * N`` so that Eq. 4 stays consistent;
4. a second (slow-EMA) state replays the same pairs, but its
   children come from its own donor centroid (D-035).

The state arrays are modified in place; :func:`revive` returns how
many clusters it revived.
"""

from __future__ import annotations

import numpy as np

from emeans._validation import check_positive_integer

# App. H.1: BF16 machine epsilon, for every input dtype (D-010).
DELTA = 2.0**-7

# Sec. 3.1: dead when N_j < tau * mean(N), with tau = 0.1.
REVIVAL_THRESHOLD = 0.1


def dead_mask(counts, threshold: float = REVIVAL_THRESHOLD) -> np.ndarray:
    """Dead clusters: ``N_j < threshold * mean(N)``.

    A uniform count vector has no dead cluster, and neither has an
    all-zero one.  ``threshold`` is ``tau`` of Sec. 3.1, in ``(0, 1]``.
    """
    counts = _check_counts(counts)
    threshold = _check_threshold(threshold)
    mean = float(counts.mean())
    if mean == 0.0:
        return np.zeros(counts.shape[0], dtype=bool)
    return counts < threshold * mean


def pick_donors(
    counts,
    n_donors: int,
    rng,
    threshold: float = REVIVAL_THRESHOLD,
    exclude=(),
) -> np.ndarray:
    """Up to ``n_donors`` distinct live clusters, probability ~ N.

    Dead clusters never donate and ``exclude`` drops donors already
    used in this event (D-013).  When fewer live clusters exist, only
    those are returned; the caller asks for ``min(#dead, #live)``.
    """
    counts = _check_counts(counts)
    n_donors = check_positive_integer("n_donors", n_donors)
    threshold = _check_threshold(threshold)
    live = np.flatnonzero(~dead_mask(counts, threshold))
    excluded = np.asarray(exclude, dtype=np.intp).ravel()
    if excluded.size:
        live = np.setdiff1d(live, excluded, assume_unique=False)
    if live.size == 0:
        return np.empty(0, dtype=np.intp)
    n_draw = min(n_donors, live.size)
    weights = counts[live]
    total = float(weights.sum())
    probabilities = None if total == 0.0 else weights / total
    chosen = rng.choice(live, size=n_draw, replace=False, p=probabilities)
    return np.asarray(chosen, dtype=np.intp)


def split_centroid(centroid, delta: float = DELTA):
    """The two Eq. 8 children of ``centroid`` (dead slot, donor slot).

    The dead child is scaled by ``[1+d, 1-d, 1+d, ...]`` and the donor
    child by the complementary pattern, with ``d = 2**-7`` (D-010).
    """
    centroid = np.asarray(centroid, dtype=np.float64)
    if centroid.ndim != 1 or centroid.size == 0:
        raise ValueError(f"centroid must be 1-D, got {centroid.shape}.")
    if not np.all(np.isfinite(centroid)):
        raise ValueError("centroid must be finite.")
    delta = _check_delta(delta)
    even = np.arange(centroid.shape[0]) % 2 == 0
    plus = np.where(even, 1.0 + delta, 1.0 - delta)
    minus = np.where(even, 1.0 - delta, 1.0 + delta)
    return centroid * plus, centroid * minus


def revive(
    states,
    rng,
    threshold: float = REVIVAL_THRESHOLD,
    delta: float = DELTA,
) -> int:
    """Revive dead clusters by splitting donors, in place (App. H.1).

    ``states`` is a sequence of ``(counts, sums)`` pairs of float64
    ndarrays: state 0 drives the decisions, the others (the slow EMA)
    replay the same pairs with their own centroids (D-035).  Returns
    the number of clusters revived.
    """
    states = list(states)
    if not states:
        raise ValueError("at least one (counts, sums) state is required.")
    checked = [_check_state(c, s, i) for i, (c, s) in enumerate(states)]
    threshold = _check_threshold(threshold)
    delta = _check_delta(delta)
    counts0 = checked[0][0]
    dead = np.flatnonzero(dead_mask(counts0, threshold))
    if dead.size == 0:
        return 0
    n_live = counts0.shape[0] - dead.size
    n_revive = min(dead.size, n_live)
    if n_revive == 0:
        return 0
    # Most starved clusters first (D-012); the rest wait for the next event.
    order = dead[np.argsort(counts0[dead], kind="stable")]
    donors = pick_donors(counts0, n_revive, rng, threshold)
    for dead_index, donor_index in zip(order[:n_revive], donors, strict=False):
        for counts, sums in checked:
            _split_into(counts, sums, int(dead_index), int(donor_index), delta)
    return int(min(n_revive, donors.size))


def _split_into(counts, sums, dead_index: int, donor_index: int, delta: float):
    """Split ``donor_index`` into the dead slot and itself, for one state."""
    donor_count = counts[donor_index]
    if donor_count <= 0.0:
        raise ValueError(f"cannot split cluster {donor_index}: {donor_count}.")
    centroid = sums[donor_index] / donor_count
    dead_child, donor_child = split_centroid(centroid, delta)
    half = donor_count / 2.0
    counts[dead_index] = half
    counts[donor_index] = half
    sums[dead_index] = dead_child * half
    sums[donor_index] = donor_child * half


def _check_counts(counts) -> np.ndarray:
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError(f"counts must be 1-D and non-empty, got {counts.shape}.")
    counts = counts.astype(np.float64, copy=False)
    if not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite.")
    if np.any(counts < 0.0):
        raise ValueError("counts must be non-negative.")
    return counts


def _check_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}.")
    return threshold


def _check_delta(delta: float) -> float:
    delta = float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    return delta


def _check_state(counts, sums, index: int):
    if not isinstance(counts, np.ndarray) or counts.dtype != np.float64:
        raise TypeError(f"state {index}: counts must be a float64 ndarray.")
    if not isinstance(sums, np.ndarray) or sums.dtype != np.float64:
        raise TypeError(f"state {index}: sums must be a float64 ndarray.")
    if counts.ndim != 1 or sums.ndim != 2:
        raise ValueError(f"state {index}: expected (k,) counts and (k, d) sums.")
    if sums.shape[0] != counts.shape[0] or counts.size == 0:
        raise ValueError(f"state {index}: inconsistent shapes.")
    if not (counts.flags.writeable and sums.flags.writeable):
        raise ValueError(f"state {index}: arrays must be writable.")
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(sums)):
        raise ValueError(f"state {index}: state must be finite.")
    if np.any(counts < 0.0):
        raise ValueError(f"state {index}: counts must be non-negative.")
    return counts, sums
