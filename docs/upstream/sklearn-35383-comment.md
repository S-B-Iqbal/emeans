# Comment for scikit-learn issue #35383 (P6.2, draft — never posted)

Target: <https://github.com/scikit-learn/scikit-learn/issues/35383> ("Add e-means clustering
(EMeans)", OPEN, 25 comments, label `Needs Decision - Include Feature`, checked 2026-11-14).
Rules: rule 01 (no write on another repository) and rule 10 (drafts only). Length target ≤ 350
words. Post only after the user's explicit go-ahead.

## Ready-to-paste text

```text
Independent data point for this discussion - not a competing proposal.

I wrote an independent implementation of e-means from the paper
(BSD-3, import `emeans`, source: github.com/S-B-Iqbal/emeans) and benchmarked it honestly on one
laptop. Caveats first: Apple M2 with *serial* kernels (no -fopenmp on Apple clang), so compare
epochs rather than seconds; synthetic Gaussian blobs; n_train = 100 000 with k = 512, inside the
range the paper's scaling laws were fitted on (B = 8192, alpha0 = 0.733). Metric =
(MSE - MSE_Lloyd@25)/MSE_Lloyd@25, median of 5 seeds, both methods at equal epoch budgets:

| dataset | epochs to Lloyd@25 | EMeans@1 | EMeans@5 | worst seed | MiniBatch@5 (tuned) |
|---|---|---|---|---|---|
| d = 96 | 1.0 (5/5 seeds) | -19.0 % | -33.9 % | -28.5 % | -47.3 % |
| d = 32 | 1.0 (4/5 seeds) | -8.3 % | -15.5 % | -13.1 % | -41.1 % |

Three readings:

1. The epochs claim reproduces here: one epoch reaches the Lloyd@25 test MSE, against 1.4-2.6 in
   Table 4. We needed 2-4 epochs at n_train = 10 000, so the automatic batch size and step do
   matter as the laws intend. At d = 96 e-means is also faster than Lloyd@25 (1.5 s vs 2.3 s).
2. The win is dimension-dependent, which matches the embedding regime described above (d ~ 30-250):
   -15.5 % at d = 32 against -33.9 % at d = 96.
3. A validation-tuned MiniBatchKMeans still wins on test MSE at every equal-epoch budget here, as
   it did at n_train = 10 000. So the strongest case is not "add another estimator" but landing
   the transferable parts in MiniBatchKMeans (cosine-annealed step with a floor; last-epoch-mean
   finalization, which gives `fit(X).labels_ == predict(X)` for free; automatic batch size and peak
   step; split-based revival instead of random reassignment) as the `algorithm="e-means"` route
   @glevv suggested.

Protocol, CSV, plots and every losing case: `benchmarks/report.md`. Parity with the existing
prototypes (thanks @cakedev0, @maxmbay): `docs/upstream/parity-matrix.md`. Our implementation
differs by supporting `sample_weight` and streaming `partial_fit` - happy to upstream either to
whichever package maintainers prefer rather than duplicate effort.
```

## Facts checked before posting (refreshed 2026-11-14, P8.1)

| Claim | Source |
|---|---|
| epochs to Lloyd@25: 1.0 at d = 96 (5/5 seeds) and 1.0 at d = 32 (4/5, one seed at 4) | `benchmarks/results/large.csv`, summarised in `benchmarks/report.md` |
| EMeans@1 / @5 vs Lloyd@25: −19.0 % / −33.9 % (d = 96) and −8.3 % / −15.5 % (d = 32); worst seeds −28.5 % and −13.1 % | same (`n_train = 100 000`, `k = 512`, median of 5 seeds) |
| tuned MiniBatchKMeans@5: −47.3 % (d = 96, 14.0 s) and −41.1 % (d = 32, 7.5 s), best on validation, reported on test | same; protocol in rule 09 and `benchmarks/report.md` |
| e-means faster than Lloyd@25 at d = 96: 1.5 s vs 2.3 s (and 0.66 s at 1 epoch) | same table, serial build |
| `B = 8 192`, `alpha0 = 0.733` at `n = 100 000`, `k = 512`, `T = 5` | `emeans/scaling_laws.auto_hyperparameters`; quoted in `report.md` |
| paper Table 4: 1.4–2.6 epochs; Table 10: 1.0–3.0 at `n/k = 1000` | `docs/PAPER_NOTES.md` |
| serial build reality | `benchmarks/report.md` environment block; D-044 (macOS), D-056 (Windows) |
| `sample_weight` / `partial_fit` absent from `sklearn-contrib-emeans` 0.3.1 | PyPI project description, fetched 2026-11-14 (`docs/upstream/parity-matrix.md`) |
| `glevv`'s `algorithm="e-means"` suggestion, `betatim`'s embedding regime, `cakedev0`/`maxmbay` prototypes | issue comments 2026-11-12/13, read 2026-11-14 |

The earlier `n_train = 10 000` numbers (2.4–3.0 epochs, −19.8 % at `d = 96`, ~2x slower than
Lloyd@25) are still in `benchmarks/report.md` under "Results below the fitted range"; they are no
longer quoted in the comment, because the in-range run supersedes them and contradicts the
wall-clock claim.

Not claimed anywhere: that e-means is a new algorithm, that our package is faster than the other
implementation, or any speedup number from the paper (their 9–15× came from 64-core Xeon with
Faiss assignment, Sec. 4.1 / Table 11 — not reproducible on a laptop).
