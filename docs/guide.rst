User guide
==========

`EMeans` clusters data with **e-means** (Bessa et al., `arXiv:2611.09501
<https://arxiv.org/abs/2611.09501>`_): a mini-batch k-means whose centroids are exponential
moving averages (EMAs) of the points assigned to them, with a cosine-annealed step that ends
on a floor.  The batch size and the peak step come from scaling laws published with the paper,
so the only parameter you have to choose is ``n_clusters``.

.. code-block:: python

   import numpy as np
   from emeans import EMeans

   X = np.random.RandomState(0).normal(size=(100_000, 96))
   model = EMeans(n_clusters=512).fit(X)
   labels, centres, inertia = model.labels_, model.cluster_centers_, model.inertia_

One epoch is a pass over the whole training set in random order; ``n_epochs=5`` (the default)
is what the paper uses for its headline tables.  ``labels_`` comes from one extra assignment
pass over the final centres, so ``model.labels_`` is always ``model.predict(X)``.

When to use it
--------------

- **Embedding-like data** (``d`` roughly 30–250) with many clusters: this is the regime the
  paper's laws were fitted on and where we measure the largest gain over Lloyd (see
  ``benchmarks/report.md`` for our own numbers, including the cases where it loses).
- **Streaming or drifting data**: the EMA state has bounded memory, so recent points dominate;
  :meth:`~emeans.EMeans.partial_fit` continues training with a constant step.
- **Weighted data**: ``sample_weight`` is accepted by :meth:`~emeans.EMeans.fit` and
  :meth:`~emeans.EMeans.partial_fit` (a capability the Faiss implementation does not have).

When *not* to use it
--------------------

- Very low-dimensional data (``d`` of 2–4) or tiny data sets (``n < 10^5``, ``n / k < 8``):
  the laws are extrapolations there.  ``EMeans`` still runs, but emits a ``UserWarning`` and
  you should compare it against :class:`~sklearn.cluster.KMeans` before trusting it.
- If you need a guaranteed Lloyd fixed point: e-means reaches the Lloyd inertia quickly but
  stops on its own schedule, not at an exact assign/update fixed point.

Choosing ``n_clusters``
-----------------------

Same rules of thumb as k-means (elbow, silhouette, or a downstream task).  The automatic
batch size and step are computed from ``n``, ``k`` and ``n_epochs``; nothing else needs tuning.
Pass ``batch_size`` and ``alpha0`` explicitly to pin the schedule for an ablation, e.g.
``alpha0=1.0, batch_size=n`` for a deterministic full-batch mean update.

Fitted attributes
-----------------

``cluster_centers_``, ``labels_``, ``inertia_``, plus ``n_iter_`` (epochs completed),
``n_steps_`` (EMA updates performed), ``batch_size_`` and ``alpha0_`` (the resolved
hyper-parameters).  ``n_steps_`` has the same meaning as in
:class:`~sklearn.cluster.MiniBatchKMeans`: minibatches processed.

Dead clusters, revival and finalization
---------------------------------------

Clusters that stop receiving points are "dead": when ``N_j < revival_threshold * mean(N)``
they are revived by splitting a donor drawn with probability proportional to its count, and
each child inherits half of it (App. H.1 of the paper).  This happens every ``revival_period``
batches (5 by default) and keeps ``k`` useful clusters alive without a second pass over the
data.

``finalization`` chooses which centres ``cluster_centers_`` returns:

- ``"last_epoch_mean"`` (default) - the (weighted) mean of the points assigned to each cluster
  during the last epoch, guaranteed to be no worse than the EMA centres on the training MSE;
- ``"slow_ema"`` - a second EMA family with a slower step (``slow_ema_speed``), never used for
  assignment, which helps test MSE at low ``n / k``;
- ``None`` - the EMA centres ``S / N`` as they are.

Reproducibility, dtypes and threads
-----------------------------------

``random_state`` drives the initialisation, the per-epoch shuffles and the revival draws, so
two runs with the same seed are identical.  ``float32`` input stays ``float32`` (the state is
accumulated in ``float64``), and CSR input is supported.

Compiled kernels (Cython, optional OpenMP) are used by default, with a NumPy fallback selected
through ``emeans._backend.set_backend("numpy" | "cython")``.  Whether a build has OpenMP is
reported by ``emeans._kernels.has_openmp()``: Linux wheels do, macOS and Windows wheels are
serial because of their toolchains.

Limitations
-----------

- The scaling laws were fitted on 96-dimensional data with ``10^5 <= n <= 10^8``,
  ``256 <= k <= 65,536`` and ``n / k >= 8``; outside that box they are extrapolations, and
  ``EMeans`` warns you when you are outside it.
- ``n_epochs`` is an integer in this version, so the paper's 0.2-epoch protocol cannot be
  reproduced exactly.
- On this machine our own benchmarks have e-means losing to a validation-tuned
  :class:`~sklearn.cluster.MiniBatchKMeans` in some configurations; the numbers, including the
  losses, are in ``benchmarks/report.md`` in the repository.

References
----------

Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou.  *e-means: outperforming k-means (in
one-tenth the epochs)*.  arXiv:2611.09501, 2026.
