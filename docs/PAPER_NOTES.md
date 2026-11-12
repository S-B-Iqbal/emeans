# Paper notes — e-means (arXiv:2611.09501v1)

Source: `docs/e-means.pdf` · HTML: https://arxiv.org/html/2611.09501v1
Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou — FAIR at Meta / ENPC — 6 Nov 2026.
These notes are a working digest in our own words. When in doubt, re-read the paper.

## Notation
X = {x_1..x_n} ⊂ R^d · k clusters · T epochs (one epoch = one pass over all n points in random
order) · B batch size · M = T·⌊n/B⌋ updates · γ decay · α = 1 − γ step.

## Eq. 1 — objective
MSE(X, C) = (1/n) Σ_i ‖x_i − c_{a(i)}‖², a(i) = argmin_j ‖x_i − c_j‖².

## Eq. 2–4 — EMA state (Sec. 3.1)
N_j ← γN_j + (1−γ)n_j ; S_j ← γS_j + (1−γ)s_j ;
n_j = #{x ∈ batch : a(x)=j}, s_j = Σ_{x∈batch, a(x)=j} x ; c_j = S_j / N_j (centroids never stored).
Init: N_j = 1e-10, S_j = N_j · c_j^(0) (so Eq. 4 returns the initial centroids exactly).

## Eq. 5 — cosine step schedule
α(t) = α0 (ε + (1−ε) ½ (1 + cos(πt/M))), ε = 0.1, t ∈ {0, …, M−1}.

## Algorithm 1 (summary)
random init from X → N=1e-10, S=N·c, t=0 → for T epochs, for each batch of a fresh shuffle:
t ← t+1; γ ← γ(t); assign with c = S/N; compute n_j, s_j; EMA update;
if t mod 5 = 0: revive dead clusters → after training: last-epoch-mean finalization.

## Dead-cluster revival (Sec. 3.1, App. H)
Every 5 batches, clusters with N_j < τ·mean(N), τ = 0.1, are rebuilt by splitting a donor
(the Faiss procedure). App. H.1: donor drawn without replacement with probability ∝ N_j';
Eq. 8: c_j := c_j' ⊙ [1+δ, 1−δ, 1+δ, …], c_j' := c_j' ⊙ [1−δ, 1+δ, 1−δ, …], δ ≪ 1 = BF16 machine
epsilon; donor count halved, each child gets N_j'/2; S rewritten as c·N. Constant cost; splits
along a hyperplane through the donor centroid. App. H.2: a cluster counts as dead when < 10 vectors
in the final assignment; resample-from-batch and "off" leave many more dead clusters (Table 14).

## Finalization (Sec. 3.2, App. I)
- last-epoch-mean: record all last-epoch assignments; centroid = mean of its assigned points.
  Default; guaranteed to lower train MSE; negligible cost.
- slow-EMA: extra EMAs with slower decay (Sec. 3.2 "half-life h"; App. I/Table 15 select speed from
  {1/16, 1/4, 1, 4} × final cosine-floor step), same splitting logic, not used for assignment.
  Improves test MSE; effect largest at low n/k. Fig. 22 explores 128× slower to 4× faster.

## Scaling laws (Sec. 3.3, Table 2, App. B)
Eq. 6: α0 = min(a_c k^a_k B^a_B n^a_n T^a_T, 1)
Eq. 7: B = b_c k^b_k n^b_n T^b_T, rounded to nearest power of two, capped at n.

| law | c | k | B | n | T | OOD % | LOSO % | R² |
|---|---|---|---|---|---|---|---|---|
| step α0 (Eq. 6) | 0.824 | −0.46 | 0.90 | −0.45 | −0.11 | 0.07 | 0.07 | 0.95 |
| batch B, knee 0.2% | 0.035 | 0.13 | — | 0.88 | 0.81 | 0.19 | 0.20 | 0.96 |
| batch B, knee / 8 | 0.0044 | 0.13 | — | 0.88 | 0.81 | 0.07 | 0.06 | n/a |

Default batch law used for n ≤ 2M; for n > 2M the law is divided by 8.
Fitted only on DEEP (d=96), validation MSE, 1e5 ≤ n ≤ 1e8, 256 ≤ k ≤ 65,536, n/k ≥ 8, ~75k runs,
3 seeds. A d-exponent was tried and dropped as unstable. Not optimal on very low-dimensional /
tiny data (App. D.2, Fig. 11); local intrinsic dimension is what matters.
App. B.2 ablations: the power law beats logit/raw/linear forms; dropping B from the step law hurts most.

## Experimental protocol (Sec. 4.1)
k from 4√n to 32√n (1M at the 1B scale) · no hyperparameter tuning · random init (uniform without
replacement) · held-out 20,000 vectors (dedup), split 50% validation / 50% test · metric:
(MSE − MSE_Lloyd@25)/MSE_Lloyd@25 · T ≤ 25 · CPU: Xeon Platinum 8462Y+ (64 cores) with Faiss for
assignment; Faiss sub-sampling (256 pts/cluster) disabled for baselines.

## Results to compare against (qualitatively)
- Table 4: epochs to Flash-KMeans@25 test MSE: 1.4–2.6; wall-clock 9.5–45.5× (GPU).
- Table 5: 3 epochs of e-means beat Flash-KMeans@25 on all 8 configs (random init).
- Fig. 3: on DEEP/ImageNet-CLIP/SPACEV/SIFT, e-means ends below Lloyd on train and test; Sculley and
  Staleness-Reduction beat Lloyd at 1 epoch but converge higher.
- Table 10 (n/k = 1000): 1.0–3.0 test epochs to Lloyd@25.
- Table 11 (VIBE, CPU 64 cores vs Faiss): 9.0–15.0× speedup; 1.6–2.6 epochs.
- App. D.2 synthetic: below Lloyd on train in all 140 runs; test worse in 10 (overfitting in
  high-d cube cells, Fig. 10).
- App. G.1: k-means++ init slightly helps but costs ~2 orders of magnitude more time.
- App. E: EMA ≫ SGD/Adam at large batches (EMA centroid stays in the convex hull).

## Baselines config (App. C, Table 9)
Sculley = scikit-learn 1.7.2 MiniBatchKMeans with max_no_improvement=None, tol=0; swept batch size
{256,…,524288} and reassignment_ratio {0, 0.001, 0.01, 0.05, 0.1, 0.5}.
