# emeans - e-means in Python

`emeans` is an independent, BSD-3-Clause implementation of **e-means**
(Bessa, Mazare, Szilvasy, Touvron, Picard, Douze, Jegou - *e-means: outperforming k-means
(in one-tenth the epochs)*, [arXiv:2611.09501](https://arxiv.org/abs/2611.09501), 6 Nov 2026)
as a scikit-learn compatible estimator, `EMeans`.

> **Status: beta.** `EMeans` fits, predicts, streams (`partial_fit`), runs compiled
> assignment and reduction kernels (OpenMP where the toolchain provides it), passes the
> scikit-learn `parametrize_with_checks` suite with two justified xfails, and ships
> reproducible benchmarks **including their losing cases**. Not yet on PyPI.

## Installation

The PyPI distribution is called **`emeans-cluster`** because `emeans` is an unrelated Rust
implementation of the same paper; the **import name is `emeans`**.

```bash
pip install emeans-cluster          # once released
```

Until then, from a source checkout (needs `meson-python`, `meson`, `ninja`, `cython`,
`numpy`, `scipy` installed first):

```bash
pip install -e . --no-build-isolation
pytest --pyargs emeans -q
```

## Quickstart

```python
import numpy as np
from emeans import EMeans

X = np.random.RandomState(0).normal(size=(20_000, 64))
model = EMeans(n_clusters=256).fit(X)   # batch size and step come from the scaling laws
print(model.n_iter_, model.n_steps_, model.inertia_)
labels = model.labels_        # identical to model.predict(X)
centers = model.cluster_centers_
```

`n_clusters` is the only knob you have to set: the batch size (Eq. 7) and the peak step
(Eq. 6) are resolved by the fitted scaling laws of Table 2, with a cosine-annealed
schedule, dead-cluster revival and a last-epoch-mean finalization.

## API

| method | notes |
|---|---|
| `fit(X, sample_weight=None)` | dense or CSR; `labels_`, `inertia_`, `cluster_centers_` |
| `partial_fit(X, sample_weight=None)` | streaming extension with a constant step |
| `predict(X)` / `fit_predict(X)` | `labels_ == predict(X)` |
| `transform(X)` / `fit_transform(X)` | Euclidean distances, dtype of `X` |
| `score(X, sample_weight=None)` | opposite of the inertia (higher is better) |
| `get_feature_names_out()` | `emeans0 ... emeansk-1` |

## Conventions worth knowing

- `n_iter_` counts epochs, `n_steps_` counts EMA updates `T * ceil(n / B)` (not the same
  meaning as in `KMeans`).
- `labels_` comes from one extra assignment pass over the final centroids, so it always
  equals `predict(X)`.
- The scaling laws were fitted on DEEP (d = 96) with `1e5 <= n <= 1e8` and `n / k >= 8`
  (Sec. 4.1, App. B.1). Outside that range a `UserWarning` is emitted: pass `batch_size`
  and `alpha0` explicitly for small or very low-dimensional data.
- OpenMP is optional: `from emeans._kernels import has_openmp, max_threads` tells you how
  this build was compiled (macOS wheels are serial, Linux wheels use OpenMP).
- `partial_fit` is an engineering extension that the paper does not evaluate: the step is
  held constant instead of being annealed over an unknown horizon.

## Benchmarks (honest)

Measured on a laptop (Apple M2, 8 cores, serial kernels), `n_train = 10 000`, median of 5
seeds, test MSE relative to Lloyd@25 (negative is better):

| dataset | k | d | epochs to Lloyd@25 | EMeans at 5 epochs | worst seed |
|---|---|---|---|---|---|
| blobs-32d | 512 | 32 | 2.4 | -3.2% | -1.3% |
| blobs-96d | 512 | 96 | 3.0 | -19.8% | -10.7% |
| high-n/k | 20 | 16 | 1.0 | +0.0% | +46.0% |
| low-d | 32 | 2 | 4.0 | -2.5% | -0.1% |

EMeans reaches the Lloyd@25 test MSE in 2-4 epochs and does best in higher dimension, but a
validation-swept MiniBatchKMeans beats it on 3 of 4 configurations at this scale, the
high-`n/k` setting is a loss, and it is ~2x slower than Lloyd@25 in wall-clock here.  Full
protocol, all configurations and the deviations from the paper are in
[`benchmarks/report.md`](benchmarks/report.md); raw data in `benchmarks/results/suite.csv`.

## Not affiliated

Independent implementation written from the paper.  Not affiliated with Meta, FAIR or the
Faiss project; no code was copied from Faiss.  The reference implementation shipped in
Faiss (`contrib/e_means`) was consulted for behaviour only, and the divergences taken are
recorded in `memory-bank/decisionLog.md`.

## Citation

Cite the paper if you use the algorithm:

```bibtex
@misc{bessa2026emeansoutperformingkmeansin,
  title={e-means: outperforming k-means (in one-tenth the epochs)},
  author={Swann Bessa and Pierre-Emmanuel Mazare and Gergely Szilvasy and Hugo Touvron and
          David Picard and Matthijs Douze and Herve Jegou},
  year={2026},
  eprint={2611.09501},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2611.09501},
}
```

## Development

```bash
pip install -e . --no-build-isolation   # editable install (kernels compile here)
pytest --pyargs emeans -q                # unit tests (includes the sklearn check suite)
pytest --doctest-modules emeans -q       # doctests
pre-commit run --all-files               # ruff + cython-lint + hygiene
python benchmarks/run_suite.py          # benchmark suite -> benchmarks/results/suite.csv
python benchmarks/plots.py              # figures -> benchmarks/plots/
# python -m sphinx -W -n -b html docs docs/_build/html      # documentation
```

## License

BSD-3-Clause - see [LICENSE](LICENSE).
