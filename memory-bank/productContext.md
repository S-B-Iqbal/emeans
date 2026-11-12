# Product context

## Problem
Lloyd k-means needs ~25 full passes. scikit-learn's MiniBatchKMeans is cheap per pass but ends at
higher inertia and needs tuning (batch_size, reassignment_ratio, tol).

## e-means in one paragraph
Mini-batch k-means where each cluster keeps an EMA of its count N and sum S (centroid = S/N),
with a cosine-annealed step α, dead clusters revived by splitting a donor chosen ∝ N, a
last-epoch-mean finalization, and batch size + peak step set by fitted scaling laws — so the
only knob the user must set is k.

## Paper claims (for expectation-setting, not to be asserted)
- Beats Lloyd@25 loss in 1.4–2.6 epochs on embedding datasets (abstract); ~2 epochs on average.
- 10–45× GPU wall-clock vs Flash-KMeans; ~9–15× CPU vs Faiss on 64 cores (App. D.3, Table 11).
- Ablation (Table 1, DEEP, n=1e6, k=16,384, T=3): no revival +2.97% train MSE; resampling-style
  revival +2.06%; no cosine +0.26%; no finalization +0.14%; k-means++ init −0.18% but ~2 orders
  of magnitude more init time (App. G.1).

## Users
Data scientists clustering embeddings (IVF coarse quantizers, codebooks, data curation) inside
sklearn pipelines.

## Non-goals (v0.x)
GPU, low-precision assignment, IVF approximate assignment, distributed training.
