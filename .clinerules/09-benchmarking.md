---
paths:
  - "benchmarks/**"
---
# 09 — Benchmarking honesty (Phase 4)

## Protocol (adapted from Sec. 4.1 at CPU scale)
- Random init for every method. Held-out set: up to 20,000 vectors removed from training,
  deduplicated, split 50/50 validation/test. Report TEST MSE primarily, train MSE secondarily.
- Reference: Lloyd@25 = `KMeans(algorithm="lloyd", init="random", n_init=1, max_iter=25, tol=0)`.
  Metric: relative MSE = (MSE − MSE_Lloyd@25) / MSE_Lloyd@25.
- MiniBatchKMeans baseline as in the paper's "Sculley" arm (App. C): `max_no_improvement=None,
  tol=0`, swept batch_size and reassignment_ratio, best chosen on VALIDATION, reported on test.
- EMeans: NO tuning (auto laws). Epoch budgets in steps of 0.2 (Table 4 methodology) to find
  epochs-to-reach-Lloyd@25.
- k in [4√n, 32√n] (Sec. 4.1); datasets: sklearn blobs, a high-n/k case, at least one low-d case,
  plus 1–2 public embedding sets (e.g. a VIBE dataset or SIFT1M subset).
- ≥ 5 seeds; report median and spread.

## Reporting
- Record CPU model, cores/threads, BLAS vendor, numpy/scipy/sklearn versions, OMP_NUM_THREADS.
- Count the extra labels_ pass (if used) in epoch/FLOP accounting, or report with and without it.
- Report every configuration run. Losing cases (low d, small n, n/k < 8) appear prominently.
- Expected: paper CPU speedups (≈ 9–15×) came from 64-core Xeon + Faiss assignment at ~1M rows;
  laptop-scale gains will be smaller. The headline metric is EPOCHS, wall-clock is secondary.
- Output: `benchmarks/report.md` + `benchmarks/plots/*.png`; raw CSV under `benchmarks/results/`.
