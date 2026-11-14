# User example sketch: clustering embeddings (P6.5, draft — never submitted)

`betatim` in #35383: *“Maybe clustering embeddings of sentences/images/what have you could make
for a nice user example then?”* and elsewhere in the thread maintainers asked for evidence in the
regime where this matters. Our own numbers say that regime is `d ≈ 30–250` with many clusters
(−19.8 % vs Lloyd@25 at `d = 96`, `k = 512`, versus −2.5 % at `d = 2`).

Target file in scikit-learn: `examples/cluster/plot_emeans_embeddings.py` (name only if an
estimator lands; otherwise `plot_minibatch_schedule.py` for the `MiniBatchKMeans` improvement).
Draft only (rule 01/10).

## What the example should show

1. One embedding-like dataset, small enough for the examples CI (`n ≈ 20 000`, `d = 128`,
   `k = 256`), plus the honest note that this is below the paper's fitted range
   (`n ≥ 1e5`).
2. Three methods on the same axes: `KMeans` (Lloyd, 25 iterations), `MiniBatchKMeans` with a
   couple of batch sizes, and e-means with the automatic batch size and step (no tuning).
3. Two plots: inertia against wall-clock (the Pareto view maintainers are already using) and
   inertia against epochs.
4. A caption that says what it does *not* prove: serial versus OpenMP kernels, `n` below the
   fitted range, and that a tuned `MiniBatchKMeans` can win at this size.

## Skeleton (sketch, not executed here)

```python
# License: BSD 3 clause
"""Clustering embeddings with a mini-batch k-means that needs no tuning.

Embedding matrices (d ~ 30-250, often many clusters) are where a scheduled EMA step
pays off: it reaches the k-means inertia in a few passes over the data.  Timings here
are single-machine and sorted by method, so read the epochs panel too.
"""
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

# 1. Embedding-like data: anisotropic blobs in 128 dimensions.
rng = np.random.RandomState(0)
n_samples, n_features, n_clusters = 20_000, 128, 256
centers = rng.normal(size=(n_clusters, n_features))
labels = rng.randint(n_clusters, size=n_samples)
X = centers[labels] + 0.6 * rng.normal(size=(n_samples, n_features))

results = {}

for name, make in [
    ("Lloyd (25 iters)", lambda: KMeans(init="random", n_init=1, max_iter=25, tol=0)),
    ("MiniBatch (1024)", lambda: MiniBatchKMeans(batch_size=1024, max_no_improvement=None, tol=0)),
    ("MiniBatch (4096)", lambda: MiniBatchKMeans(batch_size=4096, max_no_improvement=None, tol=0)),
    # "EMeans", lambda: EMeans(n_clusters=n_clusters),   # once the estimator lands
]:
    start = time.perf_counter()
    model = make().set_params(n_clusters=n_clusters).fit(X)
    results[name] = (model.inertia_, time.perf_counter() - start, model.n_iter_)

fig, (ax_time, ax_epochs) = plt.subplots(1, 2, figsize=(10, 4))
for name, (inertia, seconds, epochs) in results.items():
    ax_time.scatter(seconds, inertia, label=name)
    ax_epochs.scatter(epochs, inertia, label=name)
ax_time.set(xlabel="fit time (s)", ylabel="inertia", title="time")
ax_epochs.set(xlabel="epochs", ylabel="inertia", title="epochs")
ax_epochs.legend()
fig.suptitle("Mini-batch k-means on 128-d embedding-like data")
plt.show()
```

## Notes for the PR/example review

- Keep runtime under ~10 s and add no new dependency (`matplotlib` and `numpy` only).
- Do not claim a speedup: label the time panel as machine-dependent, and put the epochs panel
  first if a single panel has to be chosen.
- If the accompanying change is the `MiniBatchKMeans` improvement rather than a new estimator,
  drop the `EMeans` line and keep the same dataset — the schedule comparison is the point.
- Cite the paper in the example docstring; do not copy text from the paper.
