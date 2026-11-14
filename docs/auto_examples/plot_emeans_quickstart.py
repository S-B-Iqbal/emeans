# SPDX-License-Identifier: BSD-3-Clause
"""
Quickstart: cluster blobs with EMeans
=====================================

e-means needs only the number of clusters: the batch size (Eq. 7) and the peak step
(Eq. 6) come from the scaling laws of the paper (Table 2), the step is annealed on a
cosine, dead clusters are revived by splitting a donor, and the final centroids are the
mean of the last-epoch assignments.  This example fits three blobs in two dimensions.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from emeans import EMeans

# %%
# Data: three well-separated blobs in 2-D.
rng = np.random.RandomState(0)
centers = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
X = np.vstack([c + 0.5 * rng.normal(size=(700, 2)) for c in centers])

# %%
# Fit.  The scaling laws are fitted for 1e5 <= n <= 1e8, so this example is an
# extrapolation: the warning is silenced here (explicit `batch_size` and
# `alpha0` remove it entirely).
warnings.filterwarnings("ignore", message="EMeans scaling laws")
model = EMeans(n_clusters=3, random_state=0).fit(X)
print(f"n_iter_ = {model.n_iter_}, n_steps_ = {model.n_steps_}")
print(f"inertia = {model.inertia_:.1f}")

# %%
# Plot the labels and the centroids.
figure, axes = plt.subplots(figsize=(5, 4))
axes.scatter(X[:, 0], X[:, 1], c=model.labels_, s=4, cmap="tab10")
axes.scatter(*model.cluster_centers_.T, marker="x", s=150, c="k", label="centroids")
axes.legend()
figure.tight_layout()
plt.show()
