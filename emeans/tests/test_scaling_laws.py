# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :mod:`emeans._scaling_laws` (Phase 1, task P1.1).

Expected values are re-derived here from the coefficients printed in Table 2 of
arXiv:2611.09501v1, so a wrong constant in the module cannot hide behind a matching
wrong expectation.
"""

import pytest

from emeans._scaling_laws import (
    BATCH_LAW_COEF,
    BATCH_LAW_COEF_LARGE,
    BATCH_LAW_K_EXPONENT,
    BATCH_LAW_N_EXPONENT,
    BATCH_LAW_T_EXPONENT,
    LARGE_N_THRESHOLD,
    STEP_LAW_B_EXPONENT,
    STEP_LAW_COEF,
    STEP_LAW_K_EXPONENT,
    STEP_LAW_N_EXPONENT,
    STEP_LAW_T_EXPONENT,
    auto_alpha0,
    auto_batch_size,
    auto_hyperparameters,
    round_to_power_of_two,
)

# (n_samples, n_clusters, n_epochs) triples used by several tests below.
LAW_CASES = [
    (10**5, 256, 5),
    (10**6, 1024, 3),
    (10**7, 65_536, 5),
    (2_000_001, 4096, 4),
]


def raw_batch_size(n_samples, n_clusters, n_epochs):
    """Eq. 7 before rounding, recomputed from the printed Table 2 coefficients."""
    coefficient = 0.0044 if n_samples > 2_000_000 else 0.035
    return coefficient * n_clusters**0.13 * n_samples**0.88 * n_epochs**0.81


def raw_alpha0(n_samples, n_clusters, n_epochs, batch_size):
    """Eq. 6 before clipping, recomputed from the printed Table 2 coefficients."""
    return (
        0.824
        * n_clusters**-0.46
        * batch_size**0.90
        * n_samples**-0.45
        * n_epochs**-0.11
    )


def test_table_2_constants_match_the_paper():
    # Batch law: rows "batch B, knee at 0.2%" and "batch B, knee / 8".
    assert (BATCH_LAW_COEF, BATCH_LAW_COEF_LARGE) == (0.035, 0.0044)
    assert (BATCH_LAW_K_EXPONENT, BATCH_LAW_N_EXPONENT, BATCH_LAW_T_EXPONENT) == (
        0.13,
        0.88,
        0.81,
    )
    # Step law: row "step alpha0, Equation 6".
    assert STEP_LAW_COEF == 0.824
    assert (STEP_LAW_K_EXPONENT, STEP_LAW_B_EXPONENT) == (-0.46, 0.90)
    assert (STEP_LAW_N_EXPONENT, STEP_LAW_T_EXPONENT) == (-0.45, -0.11)
    # Sec. 3.3: the knee law is the default "for n <= 2M".
    assert LARGE_N_THRESHOLD == 2_000_000


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.25, 1),
        (0.999, 1),
        (1.0, 1),
        (1.4, 1),
        (2.0, 2),
        (2.8, 2),
        (2.9, 4),  # the geometric midpoint of 2 and 4 is 2.83
        (3.0, 4),
        (5.6, 4),
        (5.7, 8),
        (1024.0, 1024),
        (10**9, 2**30),
    ],
)
def test_round_to_power_of_two_values(value, expected):
    assert round_to_power_of_two(value) == expected


def test_round_to_power_of_two_ties_round_up():
    for exponent in range(0, 12):
        value = 2 ** (exponent + 0.5)  # exact tie between 2**e and 2**(e+1)
        assert round_to_power_of_two(value) == 2 ** (exponent + 1)


def test_round_to_power_of_two_rejects_non_finite():
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            round_to_power_of_two(bad)


def test_auto_batch_size_reproduces_the_fitted_law():
    for n_samples, n_clusters, n_epochs in LAW_CASES:
        raw = raw_batch_size(n_samples, n_clusters, n_epochs)
        expected = min(round_to_power_of_two(raw), n_samples)
        got = auto_batch_size(n_samples, n_clusters, n_epochs)
        assert got == expected


def test_auto_batch_size_switches_to_the_eighth_coefficient_above_2e6():
    at_threshold = auto_batch_size(2_000_000, 1024, 5)
    above_threshold = auto_batch_size(2_000_001, 1024, 5)
    # Neither value is capped at n, and the "/8" law differs by exactly a factor 8,
    # which survives rounding to a power of two (an integer shift of log2).
    assert above_threshold == at_threshold // 8
    raw = raw_batch_size(2_000_001, 1024, 5)
    assert above_threshold == round_to_power_of_two(raw)


def test_auto_batch_size_is_capped_at_n_samples():
    # Here the unrounded law exceeds n_samples, so the cap is what is exercised.
    assert round_to_power_of_two(raw_batch_size(1000, 100, 100)) > 1000
    assert auto_batch_size(1000, 100, 100) == 1000


def test_auto_batch_size_is_never_below_one():
    assert raw_batch_size(10, 2, 1) < 1
    assert auto_batch_size(10, 2, 1) == 1


def test_auto_alpha0_uses_the_resolved_batch_size():
    # k and n keep alpha0 below the clip at 1.0, otherwise the clipping would hide any
    # difference between the batch sizes that could be fed in.
    n_samples, n_clusters, n_epochs = 10**6, 16_384, 5
    resolved = auto_batch_size(n_samples, n_clusters, n_epochs)
    # The law is applied to the rounded value, so rounding really changed it here.
    assert resolved != raw_batch_size(n_samples, n_clusters, n_epochs)
    alpha0 = auto_alpha0(n_samples, n_clusters, n_epochs, resolved)
    assert alpha0 < 1.0
    assert alpha0 == pytest.approx(
        raw_alpha0(n_samples, n_clusters, n_epochs, resolved), rel=1e-12
    )
    # Feeding the unrounded batch size instead would change the peak step (D-006).
    unrounded = raw_batch_size(n_samples, n_clusters, n_epochs)
    assert raw_alpha0(n_samples, n_clusters, n_epochs, unrounded) != pytest.approx(
        alpha0, rel=1e-3
    )


def test_auto_alpha0_is_clipped_at_one():
    alpha0 = auto_alpha0(10**5, 2, 1, 10**5)
    assert raw_alpha0(10**5, 2, 1, 10**5) > 1
    assert alpha0 == 1.0


def test_auto_alpha0_is_monotone_in_its_four_arguments():
    base = {"n_samples": 10**6, "n_clusters": 1024, "n_epochs": 5, "batch_size": 8192}
    reference = auto_alpha0(**base)
    assert auto_alpha0(**{**base, "batch_size": 4096}) < reference
    assert auto_alpha0(**{**base, "n_samples": 10**7}) < reference
    assert auto_alpha0(**{**base, "n_clusters": 4096}) < reference
    assert auto_alpha0(**{**base, "n_epochs": 10}) < reference


def test_auto_hyperparameters_returns_the_resolved_pair():
    batch_size, alpha0 = auto_hyperparameters(10**6, 1024, 5)
    assert batch_size == auto_batch_size(10**6, 1024, 5)
    assert alpha0 == auto_alpha0(10**6, 1024, 5, batch_size)
    assert 0.0 < alpha0 <= 1.0


def test_invalid_inputs_raise():
    for bad in (1.5, float("nan"), True, "10", None):
        with pytest.raises(TypeError):
            auto_batch_size(bad, 10, 5)
        with pytest.raises(TypeError):
            auto_alpha0(100, 10, 5, bad)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            auto_batch_size(bad, 10, 5)
        with pytest.raises(ValueError):
            auto_alpha0(100, 10, 5, bad)
