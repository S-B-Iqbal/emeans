# SPDX-License-Identifier: BSD-3-Clause
"""Build-level checks for the compiled kernels (Phase 3, task P3.1)."""

from emeans._kernels import _IMPORT_ERROR, HAS_KERNELS, has_openmp, max_threads


def test_compiled_kernels_are_available():
    # Every install path of this project builds the extension (wheel and editable).
    assert HAS_KERNELS is True, _IMPORT_ERROR


def test_probe_reports_a_boolean_and_a_thread_count():
    assert isinstance(has_openmp(), bool)
    assert max_threads() >= 1


def test_thread_count_matches_the_openmp_flag():
    if has_openmp():
        assert max_threads() >= 1
    else:
        assert max_threads() == 1
