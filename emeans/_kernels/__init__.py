# SPDX-License-Identifier: BSD-3-Clause
"""Compiled kernels (Phase 3)."""

try:
    from emeans._kernels._probe import has_openmp, max_threads

    HAS_KERNELS = True
    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - pure-Python install
    HAS_KERNELS = False
    _IMPORT_ERROR = exc

    def has_openmp():
        """False when the compiled kernels are not available."""
        return False

    def max_threads():
        """1 when the compiled kernels are not available."""
        return 1


__all__ = ["HAS_KERNELS", "has_openmp", "max_threads"]
