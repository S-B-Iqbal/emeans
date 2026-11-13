"""e-means: mini-batch k-means with exponential-moving-average centroids.

``emeans`` is an independent, scikit-learn compatible implementation of the
e-means algorithm (Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou,
"e-means: outperforming k-means (in one-tenth the epochs)", arXiv:2611.09501v1,
6 Nov 2026).  It is not affiliated with Meta or with the Faiss project.

The public estimator :class:`emeans.EMeans` is added in Phase 1; this module
currently only exposes the package version.

Examples
--------
>>> import emeans
>>> emeans.__version__.count(".") >= 1
True
"""

# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from emeans._emeans import EMeans

try:  # installed (wheel or editable install)
    __version__ = _distribution_version("emeans")
except PackageNotFoundError:  # pragma: no cover - running from a bare source tree
    __version__ = "0.1.0.dev0"

__all__ = ["EMeans", "__version__"]
