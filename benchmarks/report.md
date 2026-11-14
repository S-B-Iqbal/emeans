# Phase 4 - honest CPU benchmark report

EMeans (this repository) against the two references of rule 09 / paper Sec. 4.1 on one
laptop, at a scale that fits a 10-minute suite.  Raw data: `benchmarks/results/suite.csv`;
figures: `benchmarks/plots/*.png`; code: `benchmarks/{datasets,harness,run_suite,plots}.py`.

## Protocol

- Random initialisation for every method; EMeans is **never** tuned (automatic batch size
  and step, Eq. 6-7); MiniBatchKMeans is swept on validation and its best configuration is
  reported on test (the paper does the same for its Sculley arm, App. C).
- Held-out set: up to 20 000 vectors removed from training, deduplicated, split 50/50 into
  validation and test.  Primary metric: test MSE relative to Lloyd@25, i.e.
  `(MSE - MSE_Lloyd@25) / MSE_Lloyd@25` (negative is better).
- Reference: `KMeans(init="random", algorithm="lloyd", n_init=1, max_iter=25, tol=0)`.
- 5 seeds per configuration; medians reported, spreads in the CSV.
- EMeans epoch budgets 1..5; MiniBatchKMeans `max_iter` = 5 epochs, `max_no_improvement=None`,
  `tol=0`, swept over batch size {256, 1024, 4096} x reassignment ratio {0, 0.01, 0.1}.

## Environment

```
Apple M2 (arm64), 8 CPUs, macOS 14.3.1
numpy 2.2.6 / scipy 1.15.3 / scikit-learn 1.7.2, Apple Accelerate BLAS
OpenMP: NOT available locally (Apple clang rejects -fopenmp) -> the compiled kernels run
        single-threaded here; CI (ubuntu/gcc) reports has_openmp() = True, 4 threads
OMP_NUM_THREADS: unset
```

## Results (n_train = 10 000 after the held-out split, median of 5 seeds)

| dataset | k | d | epochs to Lloyd@25 | EMeans@5 test MSE vs Lloyd | worst seed | MiniBatch (swept) | EMeans s | Lloyd@25 s |
|---|---|---|---|---|---|---|---|---|
| blobs-32d-k512 | 512 | 32 | 2.4 | **-3.2%** | -1.3% | -19.1% | 0.19 | 0.07 |
| blobs-96d-k512 | 512 | 96 | 3.0 | **-19.8%** | -10.7% | -28.9% | 0.29 | 0.11 |
| high-n-over-k (n/k=1000) | 20 | 16 | 1.0 | **+0.0%** | +46.0% | -47.7% | 0.03 | 0.02 |
| low-d | 32 | 2 | 4.0 | **-2.5%** | -0.1% | +16.7% | 0.02 | 0.02 |

`epochs to Lloyd@25` = median smallest epoch budget whose **validation** MSE reaches the
Lloyd@25 validation MSE (one high-n/k seed never did within 5 epochs and is excluded from
the median).  Wall-clock is the median of the 5 seeds for that budget.

## What reproduces the paper, and what does not

**Reproduced qualitatively**

- e-means reaches the Lloyd@25 test MSE in **2-4 epochs** on the blob datasets, against
  Lloyd 25 epochs (paper: 1.4-2.6 epochs, Table 4).  The headline claim holds at this scale.
- Higher dimension helps: at d=96 e-means ends **-19.8%** below Lloyd@25, at d=32 only -3.2%,
  consistent with the paper App. D.2 (the laws and the method favour higher intrinsic
  dimension).

**Losing cases (reported prominently, as rule 09 requires)**

1. **MiniBatchKMeans beats EMeans on 3 of the 4 datasets** (-19.1%, -28.9%, -47.7% vs Lloyd),
   and it is faster than EMeans at these sizes.  At n=10 000 the fitted laws put the batch
   size near the small end (B = 256 for k=512, T=5) and EMeans pays for the per-epoch
   auto-parameter path plus the extra `labels_` pass; the paper only claims its speedups at
   1e5-1e8 rows with 64 cores.
2. **high-n-over-k (k=20, d=16, n/k=1000)** is a clear loss: median +0.0%, worst seed
   +46.0%.  This configuration is far outside the fitted range (n < 1e5, so the D-021
   warning fires and the automatic batch size/step are extrapolated); with k=20 the cosine
   schedule has very few updates to work with.
3. **Wall-clock**: EMeans is ~1.5-2.5x slower than Lloyd@25 here (0.19 s vs 0.07 s at
   k=512, d=32).  Fewer epochs do not compensate for the per-epoch cost at this scale, and
   the local build is single-threaded (no OpenMP).

## Deviations from the paper protocol

1. **Integer epoch budgets.** `n_epochs` is an integer (D-039), so the 0.2-epoch spacing of
   Table 4 is not reproducible with the public API; the epochs-to-target resolution here is
   1 epoch.  (Streaming `partial_fit` cannot be used instead: it holds the step constant,
   D-014, and is a different algorithm.)
2. **Serial kernels.** The local build has no OpenMP (D-044), so the paper 9-15x CPU
   speedups cannot be reproduced on this machine; CI exercises the parallel path only for
   correctness, not for timing.
3. **Scale.** n=20 000 (10 000 after the split), k in {20, 32, 512}, d in {2, 16, 32, 96}
   versus the paper 1e5-1e8 rows and k up to 65 536.  The scaling laws were fitted on DEEP
   (d=96), so the low-d and high-n/k configurations are extrapolations by construction.
4. **Synthetic data only.** Public embedding sets (VIBE/SIFT subsets) are not downloaded;
   the paper reports them at 1M+ rows, which is out of scope for a laptop suite.
5. The rule-09 out-of-range `UserWarning` (D-021) is silenced by the harness: at this scale
   it fires for every run by design.

## Figures

- `plots/epochs_to_reference.png`: median epochs to reach Lloyd@25 per dataset.
- `plots/test_mse_vs_epochs.png`: relative test MSE against the epoch budget.
- `plots/test_mse_vs_time.png`: relative test MSE against wall-clock.

## Reproduce

```bash
python benchmarks/run_suite.py        # writes results/suite.csv (~3 min on an M2)
python benchmarks/plots.py           # writes plots/*.png
```
