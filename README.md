# emeans — e-means in Python

`emeans` is an independent, BSD-3-Clause implementation of **e-means**
(Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou — *“e-means: outperforming k-means
(in one-tenth the epochs)”*, [arXiv:2611.09501](https://arxiv.org/abs/2611.09501), 6 Nov 2026)
as a scikit-learn compatible estimator, `EMeans`.

> **Status: pre-alpha.** The package skeleton exists, installs and is covered by CI; the
> estimator itself lands in Phase 1. Nothing here is usable for clustering yet.

## What the algorithm is

Mini-batch k-means in which each cluster keeps an exponential moving average of its count `N`
and sum `S` (the centroid is `S / N`), driven by a cosine-annealed step, with dead clusters
revived by splitting a donor cluster, a last-epoch-mean finalisation, and two fitted scaling
laws that set the batch size and the peak step automatically — so the only knob the user has
to set is the number of clusters `k`.

Paper claims (expectation-setting, not asserted here): reaching Lloyd@25 loss in 1.4–2.6
epochs on embedding datasets, 10–45× GPU / 9–15× CPU wall-clock on the authors' hardware.

## Installation

Not on PyPI yet. From a source checkout:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . --no-build-isolation
pytest --pyargs emeans -q
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Plan: layout, build system, decisions | done |
| 1 | Pure NumPy reference implementation + unit tests | in progress |
| 2 | scikit-learn compliance: `check_estimator`, CSR, `sample_weight` | planned |
| 3 | Cython + OpenMP kernels, NumPy == Cython equality tests | planned |
| 4 | Honest CPU benchmarks vs KMeans (lloyd, 25 it) and MiniBatchKMeans | planned |
| 5 | Packaging, docs, CI matrix, wheels, PyPI | planned |
| 6 | Upstream drafts: scikit-learn-contrib, upstream discussion | planned |

## Not affiliated

This is an independent implementation written from the paper. It is **not** affiliated with
Meta, FAIR, or the Faiss project, and it contains no code copied from Faiss. The reference
implementation shipped in Faiss (`contrib/e_means`) was consulted for *behaviour* only
(documented divergences are recorded in the repository's decision log).

## Citation

If you use the algorithm, cite the paper:

```bibtex
@misc{bessa2026emeansoutperformingkmeansin,
  title={e-means: outperforming k-means (in one-tenth the epochs)},
  author={Swann Bessa and Pierre-Emmanuel Mazaré and Gergely Szilvasy and Hugo Touvron and
          David Picard and Matthijs Douze and Hervé Jégou},
  year={2026},
  eprint={2611.09501},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2611.09501},
}
```

## License

BSD-3-Clause — see [LICENSE](LICENSE).
