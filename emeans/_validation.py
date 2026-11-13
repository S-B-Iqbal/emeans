# SPDX-License-Identifier: BSD-3-Clause
"""Small shared validators for the pure-math helpers.

These helpers are deliberately independent of the estimator and of sklearn so that
:mod:`emeans._scaling_laws` and :mod:`emeans._schedule` stay unit-testable on their own.
The estimator itself validates its parameters through ``_parameter_constraints``.
"""

from __future__ import annotations

from numbers import Integral


def check_positive_integer(name: str, value: int) -> int:
    """Return ``value`` as a built-in ``int``, requiring ``value >= 1``.

    ``bool`` is rejected although it is an ``int`` subclass, because ``True`` as a count
    or a batch size is always a bug.

    Raises
    ------
    TypeError
        If ``value`` is not an integer (``bool`` included).
    ValueError
        If ``value < 1``.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}: {value!r}."
        )
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value
