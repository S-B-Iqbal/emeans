# SPDX-License-Identifier: BSD-3-Clause
"""Constants for the compiled kernels."""


# Rows per chunk, mirroring ``sklearn/cluster/_k_means_common.pyx`` (line 17):
# chunks of 256 rows keep the ``(chunk, n_clusters)`` distance buffer bounded.
CHUNK_SIZE = 256
