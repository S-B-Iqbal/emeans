# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :mod:`emeans._revival` (Phase 1, task P1.4)."""

import numpy as np
import pytest

from emeans._revival import (
    DELTA,
    REVIVAL_THRESHOLD,
    dead_mask,
    pick_donors,
    revive,
    split_centroid,
)

# Eq. 8 perturbation patterns, and a reference donor centroid [1, 2].
EQ8_DEAD = np.array([1.0 + 2.0**-7, 1.0 - 2.0**-7])
EQ8_DONOR = np.array([1.0 - 2.0**-7, 1.0 + 2.0**-7])
C12 = np.array([1.0, 2.0])


def state(counts, sums):
    """A (counts, sums) pair as float64 ndarrays."""
    return [
        np.array(counts, dtype=np.float64),
        np.array(sums, dtype=np.float64),
    ]


def test_constants_match_the_paper():
    assert DELTA == 2.0**-7 == 0.0078125  # App. H.1, BF16 epsilon
    assert REVIVAL_THRESHOLD == 0.1  # Sec. 3.1


def test_split_centroid_matches_eq_8():
    dead_child, donor_child = split_centroid(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(dead_child, EQ8_DEAD * C12)
    np.testing.assert_array_equal(donor_child, EQ8_DONOR * C12)


def test_split_centroid_alternates_signs():
    dead_child, donor_child = split_centroid(np.ones(4))
    expected_dead = np.array([1, -1, 1, -1]) * DELTA + 1.0
    expected_donor = np.array([-1, 1, -1, 1]) * DELTA + 1.0
    np.testing.assert_array_equal(dead_child, expected_dead)
    np.testing.assert_array_equal(donor_child, expected_donor)


def test_split_centroid_of_a_zero_vector_stays_zero():
    dead_child, donor_child = split_centroid(np.zeros(3))
    np.testing.assert_array_equal(dead_child, np.zeros(3))
    np.testing.assert_array_equal(donor_child, np.zeros(3))


def test_dead_mask_uses_the_mean_threshold():
    mask = dead_mask(np.array([10.0, 10.0, 10.0, 0.1]))
    # mean = 7.575, bound = 0.7575: only the 0.1 is dead
    np.testing.assert_array_equal(mask, [False, False, False, True])
    np.testing.assert_array_equal(dead_mask(np.array([5.0, 5.0])), [False] * 2)
    np.testing.assert_array_equal(dead_mask(np.zeros(3)), [False] * 3)


def test_dead_mask_rejects_bad_inputs():
    with pytest.raises(ValueError):
        dead_mask(np.ones(3), threshold=0.0)
    with pytest.raises(ValueError):
        dead_mask(np.ones(3), threshold=1.5)
    with pytest.raises(ValueError):
        dead_mask(np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        dead_mask(np.array([]))


def test_pick_donors_skips_dead_and_excluded_clusters():
    counts = np.array([0.5, 10.0, 12.0, 8.0])
    rng = np.random.RandomState(0)
    donors = pick_donors(counts, 2, rng, exclude=[1])
    assert donors.size == 2
    assert len(set(donors.tolist())) == 2
    assert set(donors.tolist()) <= {2, 3}
    assert pick_donors(counts, 99, rng).size == 3  # live clusters only


def test_pick_donors_is_proportional_to_counts():
    counts = np.array([20.0, 40.0, 60.0])
    rng = np.random.RandomState(123)
    draws = np.bincount([pick_donors(counts, 1, rng)[0] for _ in range(6000)])
    frequencies = draws / draws.sum()
    np.testing.assert_allclose(frequencies, counts / counts.sum(), atol=0.03)


def test_revive_conserves_the_count_and_rewrites_the_sums():
    fast = state([0.0, 10.0], [[0.0, 0.0], [10.0, 20.0]])  # donor c = [1, 2]
    assert revive([fast], np.random.RandomState(0)) == 1
    assert fast[0].sum() == 10.0  # exact: halving is exact in binary
    assert fast[0][0] == fast[0][1] == 5.0
    np.testing.assert_allclose(fast[1][0] / fast[0][0], EQ8_DEAD * C12)
    np.testing.assert_allclose(fast[1][1] / fast[0][1], EQ8_DONOR * C12)
    np.testing.assert_allclose(fast[1][0], EQ8_DEAD * C12 * 5.0)


def test_revive_leaves_a_healthy_state_untouched():
    fast = state([5.0, 5.0, 5.0], np.ones((3, 2)))
    before = [array.copy() for array in fast]
    assert revive([fast], np.random.RandomState(0)) == 0
    np.testing.assert_array_equal(fast[0], before[0])
    np.testing.assert_array_equal(fast[1], before[1])


def test_revive_does_not_reuse_a_donor_in_one_event():
    counts = [0.0, 0.0, 10.0, 10.0, 10.0]
    sums = [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
    fast = state(counts, sums)
    assert revive([fast], np.random.RandomState(3)) == 2
    assert fast[0].sum() == 30.0
    assert np.count_nonzero(fast[0]) == 5  # two revived, two split donors, one intact


def test_revive_fills_only_as_many_slots_as_there_are_live_clusters():
    fast = state([0.0, 0.0, 0.0, 10.0], np.vstack([np.zeros((3, 2)), [[10.0, 10.0]]]))
    assert revive([fast], np.random.RandomState(0)) == 1
    assert (fast[0] == 0.0).sum() == 2  # two dead clusters wait for next event
    assert fast[0].sum() == 10.0


def test_revive_splits_the_slow_state_from_its_own_centroid():
    fast = state([0.0, 10.0], [[0.0, 0.0], [10.0, 20.0]])  # donor c = [1, 2]
    slow = state([0.0, 30.0], [[0.0, 0.0], [90.0, 90.0]])  # donor c = [3, 3]
    assert revive([fast, slow], np.random.RandomState(0)) == 1
    np.testing.assert_allclose(slow[1][0] / slow[0][0], EQ8_DEAD * 3.0)
    np.testing.assert_allclose(slow[1][1] / slow[0][1], EQ8_DONOR * 3.0)
    assert slow[0][0] == slow[0][1] == 15.0
    np.testing.assert_allclose(fast[1][0] / fast[0][0], EQ8_DEAD * C12)


def test_revive_is_deterministic_for_a_fixed_seed():
    def run():
        fast = state([0.0, 0.0, 9.0, 11.0], np.eye(4, 2) * 10.0)
        revive([fast], np.random.RandomState(7))
        return fast[0].copy(), fast[1].copy()

    first, second = run(), run()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_revive_requires_float64_and_writable_states():
    counts = np.array([0.0, 1.0])
    sums = np.zeros((2, 1))
    rng = np.random.RandomState(0)
    with pytest.raises(TypeError):
        revive([(counts.astype(np.float32), sums.astype(np.float32))], rng)
    readonly = sums.copy()
    readonly.setflags(write=False)
    with pytest.raises(ValueError):
        revive([(counts, readonly)], np.random.RandomState(0))
    with pytest.raises(ValueError):
        revive([(counts, np.zeros((3, 1)))], np.random.RandomState(0))
    with pytest.raises(ValueError):
        revive([], np.random.RandomState(0))


def test_pick_donors_returns_empty_when_all_live_are_excluded():
    counts = np.array([0.0, 10.0])  # cluster 0 is dead (mean 5, bound 0.5)
    assert pick_donors(counts, 1, np.random.RandomState(0), exclude=[1]).size == 0


def test_pick_donors_validates_n_donors():
    with pytest.raises(ValueError):
        pick_donors(np.ones(3), 0, np.random.RandomState(0))
    with pytest.raises(TypeError):
        pick_donors(np.ones(3), 1.5, np.random.RandomState(0))


def test_split_centroid_validates_its_inputs():
    with pytest.raises(ValueError):
        split_centroid(np.ones((2, 2)))
    with pytest.raises(ValueError):
        split_centroid(np.zeros(0))
    with pytest.raises(ValueError):
        split_centroid(np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        split_centroid(np.ones(2), delta=0.0)
