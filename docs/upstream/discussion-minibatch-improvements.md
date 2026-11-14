# Discussion: improvements to `MiniBatchKMeans` (P6.3, draft — never posted)

Target: a scikit-learn Discussion (or a follow-up comment in #35383) framed under the FAQ clause
"a clear-cut improvement to a widely-used method". Draft only (rule 01 / rule 10).

## Why this framing

The FAQ inclusion bar (≥ 3 years since publication, ≥ 200 citations) rules out a new-estimator PR
for a paper published on 6 Nov 2026, and #35383 already carries the label
`Needs Decision - Include Feature` with `betatim` asking for consensus and evidence rather than
speed of execution. What the thread is short of is not another estimator but a measured answer to
"is any of this transferable to what we already ship?". Our own data says the relevant parts are
the *step schedule* and the *finalization*, not the estimator.

## Four proposals, each independently landable

| # | Change | What it replaces | Back-compat |
|---|---|---|---|
| 1 | Optional **cosine-annealed step with a floor**: global step α(t) = α0·(ε + (1−ε)·½(1+cos(πt/M))) with `M = max_iter · n_batches`, ε = 0.1 | today's update is an **unbounded-count running mean**: `center_j ← (center_j · count_j + Σ_{batch→j} x)/(count_j + w_j)`, so the effective step α_j = w_j/(count_j + w_j) is cluster-dependent, is not a schedule, has no floor, and the final centre depends on the batch order (`_k_means_minibatch.pyx`, `update_center_dense`) | new parameter, default keeps today's running mean |
| 2 | **Last-epoch-mean finalization**: record the assignments of the final epoch and return their per-cluster (weighted) mean | returning the EMA centroids of the final batch | new parameter; the mean is provably no worse than the EMA centroids on *train* MSE, and with `compute_labels=True` the labels pass becomes free (`fit(X).labels_ == predict(X)` by construction) |
| 3 | **Automatic batch size and peak step** (`batch_size="auto"`, `alpha0="auto"`) from the paper's Table 2 laws, power-of-two rounding, cap at `n` | manual default `batch_size=1024` | defaults unchanged; the auto path warns when `n`, `n/k` or `d` fall outside the range the laws were fitted on |
| 4 | **Split-based revival** for under-used clusters: split the largest donor along the BF16-epsilon pattern (Eq. 8) instead of re-drawing random samples from the batch | `reassignment_ratio`: `to_reassign = counts < reassignment_ratio * counts.max()`, at most `0.5 · batch_size` clusters, new centres drawn **uniformly at random from the batch** (`_kmeans.py`, `_mini_batch_step`) | new parameter (e.g. `revival="off" | "split"`), default `"off"` |

Proposals 1 and 2 are the ones with measured support; 3 is a convenience whose benefit is
invisible below ~1e5 rows; 4 addresses dead clusters that `reassignment_ratio` handles poorly on
skewed data (paper App. H.2, Table 14), but we have not measured it against
`reassignment_ratio` on sklearn's own benchmarks — that would be part of the PR discussion, not
something to assert.

## Evidence to quote (and its limits)

- Independent implementation, `n_train = 10 000`, median of 5 seeds, test MSE relative to
  `KMeans(algorithm="lloyd", init="random", n_init=1, max_iter=25)`:
  2.4–3.0 epochs to match Lloyd@25 at `d = 32/96`, **−19.8 %** at `d = 96` after 5 epochs;
  `+0.0 %` and `+46 %` (worst seed) in the high-`n/k` case.
- Limits to state: laptop, **serial** kernels (OpenMP only on Linux here), `n` below the range the
  laws were fitted on, and a tuned `MiniBatchKMeans` still wins on 3 of our 4 configurations at
  that size.
- Therefore the honest pitch is: *annealing + last-epoch-mean raise the floor of a widely used
  estimator; the auto laws are only worth adding if the maintainers want the "no tuning" story
  for large `n`.*

## Questions for maintainers

1. Is a parameter that changes `MiniBatchKMeans`'s convergence floor acceptable, or should the
   whole thing live behind `algorithm=`?
2. Which of 1–4 would be reviewed as a standalone PR, and in what order?
3. Is there an existing benchmark (`sklearn/benchmarks` or the PR dashboard) we should extend
   instead of adding another one?

## Verification required before posting

- [x] Read `sklearn/cluster/_kmeans.py` and `_k_means_minibatch.pyx` in the read-only clone
      (`_ref/scikit-learn`, 2026-11-14): confirmed the count-based running mean
      (`update_center_dense`: undo `weight_sums` scaling, add the batch, divide by the new
      `weight_sums`), `_mini_batch_step(..., random_reassign, reassignment_ratio)` choosing new
      centres uniformly from the batch, `self._counts` as `weight_sums`, and
      `_mini_batch_convergence` with `max_no_improvement`.
- [ ] Re-check the same lines against `main` on the day of posting, and confirm the `max_iter`
      loop semantics (`n_batches`) so `M = max_iter · n_batches` is exact.
- [ ] Re-run the two headline configurations with `OMP_NUM_THREADS=1` on both methods so the
      comparison is fair under a constant thread count.
- [ ] Turn the four items into our own prose in the posted message (no copied sklearn text).
