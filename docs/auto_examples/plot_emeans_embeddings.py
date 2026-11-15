# SPDX-License-Identifier: BSD-3-Clause
"""
Clustering embeddings with EMeans
=================================

Embedding matrices are usually high-dimensional (``d`` around 30-250) and are often
clustered into many groups - the regime the paper's scaling laws were fitted on, and
where the method is meant to pay off.  This example compares
:class:`~emeans.EMeans` with :class:`~sklearn.cluster.KMeans` and
:class:`~sklearn.cluster.MiniBatchKMeans` on a synthetic embedding-like matrix, in
*passes over the data* and in *wall-clock*.

``EMeans`` needs no tuning beyond the number of clusters: the batch size and the peak
step come from the laws of the paper (Eq. 6-7, Table 2), and the step is annealed on a
cosine.

Two honest caveats, both visible in the output below:

- this run is smaller than the range the laws were fitted on (``n_train >= 1e5``), so
  ``EMeans`` emits its extrapolation warning and the parameters are extrapolations;
- at this size a ``MiniBatchKMeans`` given the same budget reaches a lower inertia.
  The paper's claim is about *epochs to reach Lloyd's inertia* on much larger,
  higher-dimensional data; the measured evidence for that regime, including the
  configurations where e-means loses, is in ``benchmarks/report.md`` of the repository.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs

from emeans import EMeans

# %%
# Embedding-like data: 128 dimensions and many clusters, then a random rotation so the
# axes are not aligned with the cluster structure (as in a real embedding).
n_samples, n_features, n_clusters = 20_000, 128, 64
X, _ = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    cluster_std=1.5,
    random_state=0,
)
rotation = np.random.RandomState(0).normal(size=(n_features, n_features))
X = X @ (rotation / np.sqrt(n_features))

# %%
# The reference is Lloyd's algorithm stopped at 25 iterations, as in the paper.
# EMeans is then run at several epoch budgets with its automatic batch size and step.
reference = KMeans(
    n_clusters=n_clusters,
    init="random",
    n_init=1,
    max_iter=25,
    tol=0,
    random_state=0,
)
start = time.perf_counter()
reference.fit(X)
reference_seconds = time.perf_counter() - start
reference_mse = reference.inertia_ / n_samples

budgets = [1, 2, 3, 5]
results = {}
with warnings.catch_warnings():
    # The data set is below the range the laws were fitted on; see the module docstring.
    warnings.filterwarnings("ignore", message="EMeans scaling laws")
    for n_epochs in budgets:
        model = EMeans(n_clusters=n_clusters, n_epochs=n_epochs, random_state=0)
        start = time.perf_counter()
        model.fit(X)
        results[n_epochs] = (
            model.inertia_ / n_samples,
            time.perf_counter() - start,
            model.n_revivals_,
        )

mini = MiniBatchKMeans(
    n_clusters=n_clusters,
    batch_size=1024,
    max_iter=3,
    n_init=1,
    max_no_improvement=None,
    tol=0,
    random_state=0,
)
start = time.perf_counter()
mini.fit(X)
mini_seconds = time.perf_counter() - start
mini_mse = mini.inertia_ / n_samples

print(f"Lloyd@25            MSE {reference_mse:7.2f}  {reference_seconds:5.2f} s")
for n_epochs, (mse, seconds, revivals) in results.items():
    share = 100.0 * (mse / reference_mse - 1.0)
    print(
        f"EMeans@{n_epochs} epochs    MSE {mse:7.2f}  {seconds:5.2f} s  "
        f"{share:+5.1f} % vs Lloyd@25, {revivals} revivals"
    )
print(f"MiniBatchKMeans@3   MSE {mini_mse:7.2f}  {mini_seconds:5.2f} s")

# %%
# Against passes over the data on the left, against wall-clock on the right.  The dashed
# line is the Lloyd@25 reference.
mse_by_epochs = [results[n][0] for n in budgets]
seconds_by_epochs = [results[n][1] for n in budgets]

figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for panel, xs in ((left, budgets), (right, seconds_by_epochs)):
    panel.axhline(reference_mse, linestyle="--", color="0.5", label="Lloyd@25")
    panel.plot(xs, mse_by_epochs, marker="o", label="EMeans")
left.scatter([3], [mini_mse], marker="s", color="C1")
left.annotate("MiniBatchKMeans (3 passes)", (3, mini_mse), fontsize=8)
right.scatter([mini_seconds], [mini_mse], marker="s", color="C1")
left.set(xlabel="passes over the data", ylabel="MSE (inertia / n)")
right.set(xlabel="fit time (s)")
left.set_title("budget in epochs")
right.set_title("budget in seconds")
left.legend(fontsize=8)
figure.tight_layout()
plt.show()
