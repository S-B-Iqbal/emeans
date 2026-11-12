# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :mod:`emeans._schedule` (Phase 1, task P1.2).

Covers Eq. 5 and the number of updates ``M`` (D-003, D-004).
"""

import math

import numpy as np
import pytest

from emeans._schedule import ALPHA_FLOOR_RATIO, alpha, alpha_schedule, n_updates


def test_n_updates_keeps_each_epoch_whole_when_batch_divides_n():
    assert n_updates(n_samples=12, batch_size=4, n_epochs=3) == 9


def test_n_updates_keeps_the_leftover_batch_of_each_epoch():
    # 13 = 3 * 4 + 1, so each epoch runs 4 batches and the tail is kept (D-004).
    assert n_updates(n_samples=13, batch_size=4, n_epochs=2) == 8


def test_n_updates_with_batch_size_above_n():
    assert n_updates(n_samples=10, batch_size=16, n_epochs=3) == 3


def test_n_updates_of_a_single_point():
    assert n_updates(n_samples=1, batch_size=1, n_epochs=1) == 1


def test_n_updates_validates_its_inputs():
    for bad in (0, -4):
        with pytest.raises(ValueError):
            n_updates(n_samples=bad, batch_size=4, n_epochs=2)
    for bad in (1.5, True, None):
        with pytest.raises(TypeError):
            n_updates(n_samples=10, batch_size=bad, n_epochs=2)


def test_alpha_last_step_is_exactly_the_floor():
    # cos(pi) is exactly -1.0, so the last update lands exactly on epsilon * alpha0.
    for n_steps, alpha0 in ((1, 0.5), (5, 0.3), (1000, 0.07)):
        assert alpha(n_steps, n_steps, alpha0) == alpha0 * ALPHA_FLOOR_RATIO


def test_alpha_first_step_matches_eq_5():
    n_steps, alpha0 = 37, 0.42
    expected = alpha0 * (
        ALPHA_FLOOR_RATIO
        + (1.0 - ALPHA_FLOOR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * 1 / n_steps))
    )
    assert alpha(1, n_steps, alpha0) == pytest.approx(expected, rel=1e-15)


def test_alpha_stays_within_the_floor_and_the_peak():
    n_steps, alpha0 = 64, 0.9
    for t in (1, 2, 33, n_steps - 1, n_steps):
        step = alpha(t, n_steps, alpha0)
        assert alpha0 * ALPHA_FLOOR_RATIO <= step <= alpha0


def test_alpha_schedule_is_monotone_non_increasing():
    for n_steps in (1, 5, 50, 1000):
        for alpha0 in (0.05, 0.3, 1.0):
            schedule = alpha_schedule(n_steps, alpha0)
            assert schedule.shape == (n_steps,)
            assert np.all(np.diff(schedule) <= 0.0)
            assert schedule.size == 1 or schedule[0] > schedule[-1]


def test_alpha_schedule_matches_alpha():
    n_steps, alpha0 = 11, 0.25
    schedule = alpha_schedule(n_steps, alpha0)
    for t in range(1, n_steps + 1):
        assert schedule[t - 1] == pytest.approx(alpha(t, n_steps, alpha0), rel=1e-15)
    assert schedule[-1] == pytest.approx(alpha0 * ALPHA_FLOOR_RATIO, rel=1e-15)


def test_alpha_validates_its_inputs():
    with pytest.raises(ValueError):
        alpha(0, 10, 0.5)  # the schedule has no step t = 0 (Alg. 1 increments first)
    with pytest.raises(ValueError):
        alpha(11, 10, 0.5)
    with pytest.raises(TypeError):
        alpha(1.5, 10, 0.5)
    with pytest.raises(TypeError):
        alpha(True, 10, 0.5)
    for bad_alpha0 in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            alpha(1, 10, bad_alpha0)
    for bad_ratio in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            alpha(1, 10, 0.5, alpha_floor_ratio=bad_ratio)
    with pytest.raises(ValueError):
        alpha_schedule(0, 0.5)
