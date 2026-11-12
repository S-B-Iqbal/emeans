# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for the package skeleton (Phase 0, task P0.4/P0.5).

The estimator itself lands in Phase 1 (task P1.5); these tests only pin the
package metadata and import surface so that the phase gate has something to
run.
"""

import emeans


def test_version_is_a_non_empty_string():
    assert isinstance(emeans.__version__, str)
    assert emeans.__version__


def test_public_namespace_is_declared():
    assert isinstance(emeans.__all__, list)
