# SPDX-License-Identifier: BSD-3-Clause
"""Fitted scaling laws for the e-means batch size and the peak step.

The laws are Eq. 6 (peak step ``alpha0``) and Eq. 7 (batch size ``B``) of Bessa et al.,
"e-means: outperforming k-means (in one-tenth the epochs)", arXiv:2611.09501v1.  The
coefficients are the ones printed in Table 2 of that paper, to three significant
figures (D-007).

The laws were fitted on DEEP (d = 96) with ``1e5 <= n <= 1e8``, ``256 <= k <= 65536``
and ``n/k >= 8`` (Sec. 4.1, App. B.1); outside that range they are an extrapolation
(App. D.2).  The estimator warns in that case (D-021) rather than silently correcting
the values.
"""

from __future__ import annotations

import math
from numbers import Integral

# Batch size, Eq. 7 with Table 2 coefficients:
# B = b_c * k^b_k * n^b_n * T^b_T, row "batch B, knee at 0.2%".
BATCH_LAW_COEF = 0.035
BATCH_LAW_K_EXPONENT = 0.13
BATCH_LAW_N_EXPONENT = 0.88
BATCH_LAW_T_EXPONENT = 0.81
# Table 2, row "batch B, knee / 8": the same law divided by 8 (Sec. 3.3).
BATCH_LAW_COEF_LARGE = 0.0044

# Peak step, Eq. 6 with Table 2 coefficients:
# alpha0 = a_c * k^a_k * B^a_B * n^a_n * T^a_T, row "step alpha0, Equation 6".
STEP_LAW_COEF = 0.824
STEP_LAW_K_EXPONENT = -0.46
STEP_LAW_B_EXPONENT = 0.90
STEP_LAW_N_EXPONENT = -0.45
STEP_LAW_T_EXPONENT = -0.11

# Sec. 3.3: the knee law is the default "for n <= 2M"; above that limit "we divide the
# batch law by 8".  The divided coefficient is its own Table 2 row, so the division is
# applied exactly once (D-006, D-027).
LARGE_N_THRESHOLD = 2_000_000


def round_to_power_of_two(value: float) -> int:
    """Round ``value`` to the nearest power of two, ties upward (D-027).

    Eq. 7 only says "rounded to the nearest power of two"; this pins the tie behaviour.
    The rounding happens in log2 space, so tie points are the geometric midpoints
    ``sqrt(2) * 2**k`` and an exact tie rounds up.  Values below 1 give 1, because a
    batch holds at least one point.

    Parameters
    ----------
    value : float
        Positive, finite value to round.

    Returns
    -------
    int
        The nearest power of two.
    """
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}.")
    if value < 1.0:
        return 1
    return int(2 ** math.floor(math.log2(value) + 0.5))


def auto_batch_size(n_samples: int, n_clusters: int, n_epochs: int) -> int:
    """Resolve the automatic batch size ``B`` (Eq. 7 with Table 2 coefficients).

    ``B`` is the fitted value rounded to the nearest power of two and capped at
    ``n_samples`` (Sec. 3.3).  Above :data:`LARGE_N_THRESHOLD` the "knee / 8"
    coefficient is used.

    Parameters
    ----------
    n_samples : int
        Number of training rows (``sample_weight`` is ignored here, D-017).
    n_clusters : int
        Number of clusters ``k``.
    n_epochs : int
        Number of epochs ``T``.

    Returns
    -------
    int
        Batch size in ``[1, n_samples]``.
    """
    n_samples = _check_positive_integer("n_samples", n_samples)
    n_clusters = _check_positive_integer("n_clusters", n_clusters)
    n_epochs = _check_positive_integer("n_epochs", n_epochs)

    if n_samples > LARGE_N_THRESHOLD:
        coefficient = BATCH_LAW_COEF_LARGE
    else:
        coefficient = BATCH_LAW_COEF
    raw = (
        coefficient
        * n_clusters**BATCH_LAW_K_EXPONENT
        * n_samples**BATCH_LAW_N_EXPONENT
        * n_epochs**BATCH_LAW_T_EXPONENT
    )
    return min(round_to_power_of_two(raw), n_samples)


def auto_alpha0(
    n_samples: int, n_clusters: int, n_epochs: int, batch_size: int
) -> float:
    """Resolve the automatic peak step ``alpha0`` (Eq. 6 with Table 2 coefficients).

    Parameters
    ----------
    n_samples : int
        Number of training rows.
    n_clusters : int
        Number of clusters ``k``.
    n_epochs : int
        Number of epochs ``T``.
    batch_size : int
        The **resolved** batch size, i.e. a power of two.  Sec. 3.3 fits the step law
        on runs that already use the batch law, so the resolved value must be passed
        here (D-006).

    Returns
    -------
    float
        Peak step in ``(0, 1]``.
    """
    n_samples = _check_positive_integer("n_samples", n_samples)
    n_clusters = _check_positive_integer("n_clusters", n_clusters)
    n_epochs = _check_positive_integer("n_epochs", n_epochs)
    batch_size = _check_positive_integer("batch_size", batch_size)

    value = (
        STEP_LAW_COEF
        * n_clusters**STEP_LAW_K_EXPONENT
        * batch_size**STEP_LAW_B_EXPONENT
        * n_samples**STEP_LAW_N_EXPONENT
        * n_epochs**STEP_LAW_T_EXPONENT
    )
    return min(value, 1.0)


def auto_hyperparameters(
    n_samples: int, n_clusters: int, n_epochs: int
) -> tuple[int, float]:
    """Resolve ``(batch_size, alpha0)``: Eq. 7 first, then Eq. 6 with that value."""
    batch_size = auto_batch_size(n_samples, n_clusters, n_epochs)
    alpha0 = auto_alpha0(n_samples, n_clusters, n_epochs, batch_size)
    return batch_size, alpha0


def _check_positive_integer(name: str, value: int) -> int:
    """Validate a positive integer law input and return it as a built-in ``int``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}: {value!r}."
        )
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value
