# Comment for scikit-learn issue #35383 (P6.2, draft — never posted)

Target: <https://github.com/scikit-learn/scikit-learn/issues/35383> ("Add e-means clustering
(EMeans)", OPEN, 25 comments, label `Needs Decision - Include Feature`, checked 2026-11-14).
Rules: rule 01 (no write on another repository) and rule 10 (drafts only). Length target ≤ 350
words. Post only after the user's explicit go-ahead.

## Ready-to-paste text

```text
Independent data point for this discussion — not a competing proposal.

While following this issue I wrote an independent implementation of e-means from the paper
(BSD-3, `pip install emeans-cluster`, import `emeans`) and benchmarked it honestly on one
laptop. Caveat first: Apple M2 with *serial* kernels (no -fopenmp on Apple clang), so compare
epochs, not seconds. n_train = 10 000, median of 5 seeds, metric = (MSE - MSE_Lloyd@25)/MSE:

| dataset | k | d | epochs to Lloyd@25 | EMeans@5 vs Lloyd | worst seed | tuned MiniBatchKMeans |
|---|---|---|---|---|---|---|
| blobs | 512 | 32 | 2.4 | -3.2 % | -1.3 % | -19.1 % |
| blobs | 512 | 96 | 3.0 | -19.8 % | -10.7 % | -28.9 % |
| high n/k = 1000 | 20 | 16 | 1.0 | +0.0 % | +46.0 % | -47.7 % |
| low-d | 32 | 2 | 4.0 | -2.5 % | -0.1 % | +16.7 % |

Two honest readings:

1. The epochs claim reproduces at this scale (paper Table 4: 1.4-2.6), and dimension is what
   matters — at d = 96 five epochs end ~20 % below Lloyd@25, at d = 2 only 2.5 %. That matches
   the regime you described above, embeddings with d ~ 30-250 and many clusters, which may be
   the most useful thing to document in an example.
2. A validation-swept MiniBatchKMeans still beats this e-means on 3 of 4 configurations at
   this size, and e-means costs about 2x Lloyd@25 in wall-clock here. So the strong case is not
   "add another estimator" but landing the transferable parts in MiniBatchKMeans (cosine-
   annealed step with a floor, last-epoch-mean finalization so `fit(X).labels_ == predict(X)` for
   free, automatic batch size / peak step, split-based revival instead of random reassignment)
   as the `algorithm="e-means"` route @glevv suggested.

Everything is reproducible: protocol, CSV and plots (losses included, no tuning of e-means)
in the repo's `benchmarks/report.md`; parity with the existing prototypes by @cakedev0 and
@maxmbay in `docs/upstream/parity-matrix.md`. Our implementation differs by supporting
`sample_weight` and streaming `partial_fit` — happy to upstream either to whichever package
maintainers prefer, rather than duplicate effort.
```

## Facts checked before posting (2026-11-14)

| Claim | Source |
|---|---|
| epochs to Lloyd@25: 2.4 / 3.0 / 1.0 / 4.0 (median of 5 seeds) | `benchmarks/results/suite.csv`, summarised in `benchmarks/report.md` |
| EMeans@5 vs Lloyd@25: −3.2 % / −19.8 % / +0.0 % / −2.5 %; worst seeds −1.3 % / −10.7 % / +46.0 % / −0.1 % | same |
| tuned MiniBatchKMeans: −19.1 % / −28.9 % / −47.7 % / +16.7 % (best on validation, reported on test) | same; protocol in rule 09 and `benchmarks/report.md` |
| wall-clock ≈ 2× Lloyd@25 | 0.19 s vs 0.07 s and 0.29 s vs 0.11 s in the report table (serial build) |
| paper Table 4: 1.4–2.6 epochs | `docs/PAPER_NOTES.md` |
| serial build reality | `benchmarks/report.md` environment block; D-044 (macOS), D-056 (Windows) |
| `sample_weight` / `partial_fit` absent from `sklearn-contrib-emeans` 0.3.1 | PyPI project description, fetched 2026-11-14 (`docs/upstream/parity-matrix.md`) |
| `glevv`'s `algorithm="e-means"` suggestion, `betatim`'s embedding regime, `cakedev0`/`maxmbay` prototypes | issue comments 2026-11-12/13, read 2026-11-14 |

Not claimed anywhere: that e-means is a new algorithm, that our package is faster than the other
implementation, or any speedup number from the paper (their 9–15× came from 64-core Xeon with
Faiss assignment, Sec. 4.1 / Table 11 — not reproducible on a laptop).
