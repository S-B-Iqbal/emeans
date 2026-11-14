# Honest CPU benchmark report (Phases 4 and 7)

Two result sets, same protocol and same machine:

- **In the fitted range** (`large.csv`, P7.1): `n_train = 100 000`, `k = 512`, `d = 32/96`.  This
  is the regime the paper's scaling laws were fitted on (`1e5 <= n <= 1e8`, `n/k >= 8`), so the
  automatic batch size and step are being used as intended (`B = 8 192`, `alpha0 = 0.733`).
- **Below the fitted range** (`suite.csv`, P4.1-P4.4): four datasets at `n_train = 10 000`
  (`B = 1 024`, `alpha0 = 0.318`).  Kept for continuity; its numbers are extrapolations.

Raw data: `benchmarks/results/{large,suite}.csv`; figures: `benchmarks/plots/<stem>/*.png`;
code: `benchmarks/{datasets,harness,run_suite,plots}.py`.

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
  `tol=0`, swept over batch size {256, 1024, 4096} x reassignment ratio {0, 0.01, 0.1}.  In the
  P7.1 run (`large`) the MiniBatchKMeans sweep is repeated at **every** epoch budget 1..5 with
  batch sizes {1024, 8192, 65536}, so the two methods can be compared at equal epochs; `suite.csv`
  keeps the original single sweep at 5 epochs.

## Environment

```
Apple M2 (arm64), 8 CPUs, macOS 14.3.1
numpy 2.2.6 / scipy 1.15.3 / scikit-learn 1.7.2, Apple Accelerate BLAS
OpenMP: NOT available locally (Apple clang rejects -fopenmp) -> the compiled kernels run
        single-threaded here; CI (ubuntu/gcc) reports has_openmp() = True, 4 threads
OMP_NUM_THREADS: unset
```

## Results inside the fitted range (`n_train = 100 000`, P7.1, median of 5 seeds)

`n_train = 100 000` after the 20 000-vector held-out split; `k = 512`; the automatic
hyper-parameters resolve to `B = 8 192` and `alpha0 = 0.733`.  Both methods are shown at
**equal epoch budgets**; test MSE is relative to Lloyd@25 (negative is better).

### `blobs-96d-n100k-k512` (`n_features = 96`)

| epochs | EMeans rel (median / worst seed) | EMeans s | MiniBatch rel (median / worst) | MiniBatch s |
|---|---|---|---|---|
| 1 | **-19.0 %** / -13.6 % | 0.66 | -36.1 % / -25.3 % | 3.96 |
| 2 | -25.8 % / -22.0 % | 0.86 | -43.5 % / -28.5 % | 7.05 |
| 3 | -30.8 % / -26.5 % | 1.10 | -46.3 % / -29.9 % | 9.32 |
| 4 | -31.1 % / -28.4 % | 1.26 | -46.9 % / -33.3 % | 11.90 |
| 5 | **-33.9 %** / -28.5 % | 1.49 | -47.3 % / -34.1 % | 13.97 |

Lloyd@25 reference: test MSE 22.95 in 2.26 s.  `epochs_to_ref` = median **1**, reached at epoch 1
by 5/5 seeds.  The sweep picked `batch_size = 1024, reassignment_ratio = 0.01` every time.

### `blobs-32d-n100k-k512` (`n_features = 32`)

| epochs | EMeans rel (median / worst seed) | EMeans s | MiniBatch rel (median / worst) | MiniBatch s |
|---|---|---|---|---|
| 1 | -8.3 % / -4.4 % | 0.49 | -29.4 % / -26.1 % | 1.67 |
| 2 | -9.3 % / -5.0 % | 0.55 | -35.3 % / -28.5 % | 3.57 |
| 3 | -14.3 % / -8.3 % | 0.76 | -38.7 % / -30.9 % | 4.63 |
| 4 | -15.8 % / -9.3 % | 0.88 | -36.6 % / -32.0 % | 5.99 |
| 5 | -15.5 % / -13.1 % | 1.02 | -41.1 % / -33.4 % | 7.45 |

Lloyd@25 reference: test MSE 5.759 in 2.18 s.  `epochs_to_ref` = median **1** (4/5 seeds at epoch
1, one seed at 4).  The sweep picked `batch_size = 1024, reassignment_ratio = 0.1` in 4/5 seeds.

### What this says

**Reproduced at this scale**

- e-means reaches the Lloyd@25 test MSE in **one epoch** in 9 of 10 runs (paper: 1.4-2.6 epochs in
  Table 4, 1.0-3.0 at `n/k = 1000` in Table 10).  The same code needed 2-4 epochs at
  `n_train = 10 000`, i.e. the automatic batch size and step behave as the laws intend.
- Dimension helps: after 5 epochs `d = 96` is **-33.9 %** below Lloyd@25 against **-15.5 %** at
  `d = 32`, consistent with App. D.2.
- e-means is now **faster than the reference too**: 0.66 s at 1 epoch and 1.49 s at 5 epochs
  against 2.26 s for Lloyd@25 at `d = 96`.  At `n_train = 10 000` we measured the opposite
  (~2x slower), so the small-scale timing result did not generalise - this one may not either.

**Still losing (including to a tuned baseline)**

1. **At equal epochs the validation-tuned MiniBatchKMeans is ahead in both datasets**
   (-36.1 % vs -19.0 % at 1 epoch and -47.3 % vs -33.9 % at 5 epochs at `d = 96`).  It is tuned
   per validation split; EMeans is never tuned.
2. **At equal time** the comparison splits by dimension: at `d = 96`, EMeans@5 gives -33.9 % in
   1.49 s where MiniBatch@1 gives -36.1 % in 3.96 s (nearly the same quality for 2.7x less time);
   at `d = 32`, MiniBatch@1 wins outright (-29.4 % in 1.67 s against our -8.3 % in 0.49 s).
3. Both datasets are synthetic Gaussian blobs with 512 clusters.  Real embeddings are not
   guaranteed to behave this way, and neither is a dataset with drift or very unbalanced clusters.

## Results below the fitted range (`n_train = 10 000`, median of 5 seeds)

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
3. **Scale.** The `suite` set is n=20 000 (10 000 after the split) with k in {20, 32, 512} and d
   in {2, 16, 32, 96}; the `large` set is n=120 000 (100 000 after the split) with k = 512 and d
   in {32, 96}.  The paper goes to 1e8 rows and k = 65 536.  The high-n/k and low-d
   configurations of `suite` remain extrapolations by construction.
4. **Synthetic data only.** Public embedding sets (VIBE/SIFT subsets) are not downloaded;
   the paper reports them at 1M+ rows, which is out of scope for a laptop suite.
5. The rule-09 out-of-range `UserWarning` (D-021) is silenced by the harness: it fires for every
   run of the `suite` set by design, and for none in `large`.

## Figures

Each result set has its own directory:

- `plots/<stem>/epochs_to_reference.png`: median epochs to reach Lloyd@25 per dataset.
- `plots/<stem>/test_mse_vs_epochs.png`: relative test MSE against the epoch budget.
- `plots/<stem>/test_mse_vs_time.png`: relative test MSE against wall-clock.

## Reproduce

```bash
python benchmarks/run_suite.py suite   # results/suite.csv: 4 datasets, n_train = 10 000 (~3 min)
python benchmarks/run_suite.py large   # results/large.csv: n_train = 100 000, P7.1 (~13 min)
python benchmarks/plots.py suite large # plots/<stem>/*.png
```
