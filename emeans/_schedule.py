# SPDX-License-Identifier: BSD-3-Clause
"""Cosine step schedule (Eq. 5) and the number of EMA updates ``M`` (Sec. 3.1)."""

from __future__ import annotations

import math

import numpy as np

from emeans._validation import check_positive_integer as _check_positive_integer

# Eq. 5 (:math:`\\varepsilon`): the step anneals down to a tenth of its peak value.
ALPHA_FLOOR_RATIO = 0.1


def n_updates(n_samples: int, batch_size: int, n_epochs: int) -> int:
    """Number of EMA updates ``M`` a full fit performs.

    Sec. 3.1 writes ``M = T * floor(n / B)``, which is exact only when the batch size
    divides the number of rows.  We keep the leftover batch of every epoch instead of
    dropping those rows, so ``M = T * ceil(n / B)`` (D-004); this is also the value of
    ``n_steps_`` after :meth:`emeans.EMeans.fit`.

    Parameters
    ----------
    n_samples : int
        Number of training rows.
    batch_size : int
        Resolved batch size ``B``.
    n_epochs : int
        Number of epochs ``T``.

    Returns
    -------
    int
        Number of updates, ``>= n_epochs``.
    """
    n_samples = _check_positive_integer("n_samples", n_samples)
    batch_size = _check_positive_integer("batch_size", batch_size)
    n_epochs = _check_positive_integer("n_epochs", n_epochs)
    n_batches = -(-n_samples // batch_size)  # ceil(n_samples / batch_size)
    return n_epochs * n_batches


def alpha(
    t: int,
    n_steps: int,
    alpha0: float,
    alpha_floor_ratio: float = ALPHA_FLOOR_RATIO,
) -> float:
    """Step size of update ``t`` (Eq. 5, with Algorithm 1's indexing).

    Alg. 1 line 5 increments ``t`` before evaluating the schedule, so ``t`` runs from
    1 to ``M``: the last update is exactly ``alpha_floor_ratio * alpha0``, the first is
    ``alpha0`` minus ``O(alpha0 / M**2)``.  Eq. 5's ``t in {0, ..., M-1}`` is the other
    convention; the difference is a one-step shift on a cosine (D-003).

    Parameters
    ----------
    t : int
        Update index, ``1 <= t <= n_steps``.
    n_steps : int
        Total number of updates ``M``.
    alpha0 : float
        Peak step, in ``(0, 1]``.
    alpha_floor_ratio : float, default=0.1
        ``epsilon`` of Eq. 5.

    Returns
    -------
    float
        Step size in ``[alpha_floor_ratio * alpha0, alpha0]``.
    """
    _check_update_index(t, n_steps)
    alpha0 = _check_step("alpha0", alpha0)
    alpha_floor_ratio = _check_step("alpha_floor_ratio", alpha_floor_ratio)
    cosine = math.cos(math.pi * t / n_steps)
    return alpha0 * (
        alpha_floor_ratio + (1.0 - alpha_floor_ratio) * 0.5 * (1.0 + cosine)
    )


def alpha_schedule(
    n_steps: int,
    alpha0: float,
    alpha_floor_ratio: float = ALPHA_FLOOR_RATIO,
) -> np.ndarray:
    """All ``n_steps`` steps of Eq. 5 as ``float64``, indexed by ``t - 1``.

    Convenience wrapper used by the tests (monotonicity) and by the Phase 4 figures; the
    fit loop calls :func:`alpha` directly.
    """
    _check_update_index(1, n_steps)
    alpha0 = _check_step("alpha0", alpha0)
    alpha_floor_ratio = _check_step("alpha_floor_ratio", alpha_floor_ratio)
    t = np.arange(1, n_steps + 1, dtype=np.float64)
    cosine = np.cos(np.pi * t / n_steps)
    return alpha0 * (
        alpha_floor_ratio + (1.0 - alpha_floor_ratio) * 0.5 * (1.0 + cosine)
    )


def _check_update_index(t: int, n_steps: int) -> None:
    """Validate ``1 <= t <= n_steps`` (the schedule has no step ``t = 0``)."""
    n_steps = _check_positive_integer("n_steps", n_steps)
    if isinstance(t, bool) or not isinstance(t, int):
        raise TypeError(f"t must be an integer, got {type(t).__name__}: {t!r}.")
    if not 1 <= t <= n_steps:
        raise ValueError(f"t must be in [1, n_steps={n_steps}], got {t}.")


def _check_step(name: str, value: float) -> float:
    """Validate a step-like value in ``(0, 1]``."""
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}.")
    return value
