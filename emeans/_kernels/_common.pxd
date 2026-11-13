# SPDX-License-Identifier: BSD-3-Clause
"""Declarations shared by the compiled kernels.

The helper below is a small re-implementation of what scikit-learn keeps in
``sklearn/cluster/_k_means_common.pxd``: private ``.pxd`` files cannot be
cimported across packages, so it is rewritten here with a citation (rule 07).
It lives in the ``.pxd`` so that every kernel can inline it.
"""

from cython cimport floating


cdef inline floating _sq_norm(const floating[::1] v) noexcept nogil:
    """Sum of squares of ``v`` (used for the centroid norms)."""
    cdef:
        Py_ssize_t i
        floating total = 0.0
    for i in range(v.shape[0]):
        total += v[i] * v[i]
    return total
