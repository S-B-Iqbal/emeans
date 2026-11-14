emeans: e-means clustering for scikit-learn
===========================================

`emeans` is an independent, BSD-3-Clause implementation of **e-means** (Bessa et al.,
*e-means: outperforming k-means (in one-tenth the epochs)*,
[arXiv:2611.09501](https://arxiv.org/abs/2611.09501)) as a scikit-learn compatible
estimator, `EMeans`.

The PyPI distribution is named `emeans-cluster` (the `emeans` name is an unrelated Rust
package); the import name is `emeans`.

- `n_clusters` is the only required choice: the batch size (Eq. 7) and the peak step
  (Eq. 6) come from scaling laws fitted in the paper (Table 2).
- Dead clusters are revived by splitting a donor cluster (App. H.1) and the returned
  centroids can be the last-epoch mean or a slow EMA (Sec. 3.2).
- Compiled assignment and reduction kernels with optional OpenMP, plus a NumPy reference
  backend; the two are checked against each other by the test suite.

.. toctree::
   :maxdepth: 2

   generated/examples/index
   guide
   api

Cite the paper when you use the algorithm; see the repository README for the full citation,
the benchmark report and the project status.
