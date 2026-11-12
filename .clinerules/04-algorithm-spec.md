# 04 — Algorithm spec (paper = truth; cite in code comments)

Full digest: `docs/PAPER_NOTES.md`. Approved resolutions: `memory-bank/decisionLog.md`.
Anything not in this file or the decision log is NOT allowed behaviour.

## Constructor parameters (all stored verbatim)
| Param | Default | Paper symbol / source |
|---|---|---|
| n_clusters | 8 | k |
| n_epochs | 5 | T (Sec. 3.1) |
| batch_size | "auto" | B, Eq. 7 + Table 2 |
| alpha0 | "auto" | α0, Eq. 6 + Table 2 |
| alpha_floor_ratio | 0.1 | ε, Eq. 5 |
| init | "random" | Alg. 1 line 2; also "k-means++" (App. G.1–G.2) or array (k, d) |
| revival_threshold | 0.1 | τ, Sec. 3.1 "Dead cluster revival" |
| revival_period | 5 | "every five batches", Alg. 1 line 10 |
| finalization | "last_epoch_mean" | Sec. 3.2; also "slow_ema", None |
| slow_ema_speed | 1.0 | App. I / Table 15 (multiplier on ε·α0) |
| random_state | None | — |
| verbose | 0 | — |

## State (Sec. 3.1)
Per cluster j: count `N_j ∈ R+` and sum `S_j ∈ R^d`. Centroids are never stored;
`c_j = S_j / N_j` (Eq. 4). Initialization: `N_j = 1e-10`, `S_j = N_j · c_j^(0)` (Sec. 3.1, Alg. 1 line 3).

## Per batch (Eq. 1–3, Alg. 1 lines 6–9)
1. `a(x) = argmin_j ||x − c_j||²` for every x in the batch (Eq. 1).
2. `n_j = Σ_{x: a(x)=j} w(x)`, `s_j = Σ_{x: a(x)=j} w(x)·x`  (weighted form of Eq. 3; w ≡ 1 recovers the paper).
3. `N_j ← γ N_j + (1−γ) n_j`, `S_j ← γ S_j + (1−γ) s_j`, with `α = 1 − γ` (Eq. 2).

## Step schedule (Eq. 5)
`α(t) = α0 · (ε + (1−ε) · ½ · (1 + cos(π t / M)))`, ε = 0.1, t ∈ {0,…,M−1}, M = T·⌊n/B⌋ (Sec. 3.1).
⚠ Alg. 1 increments t BEFORE computing γ(t) (t starts at 1). See Q-001 / Q-002 for the indexing
and remainder-batch decisions; do not resolve silently.

## Dead-cluster revival (Sec. 3.1, App. H.1)
Every `revival_period` batches (Alg. 1: `t mod 5 = 0`):
1. dead = `{ j : N_j < τ · (1/k) Σ_i N_i }`.
2. For each dead j: draw donor j' WITHOUT replacement, probability ∝ `N_j'`.
3. Children: `c_j := c_j' ⊙ [1+δ, 1−δ, 1+δ, …]`, `c_j' := c_j' ⊙ [1−δ, 1+δ, 1−δ, …]` (Eq. 8),
   δ = machine epsilon of BF16 = 2⁻⁷ = 0.0078125.
4. `N_j = N_j' = N_j'_old / 2`; rewrite `S = c · N` for both (keeps Eq. 4 consistent).
Invariants to test: Σ N unchanged by revival; `S_j == c_j · N_j` exactly after split;
donor frequency ∝ N; no donor reused within one event.

## Epochs
Fresh random permutation of X every epoch (Sec. 3.1 "single pass over all n training points in
random order"; Alg. 1 line 5).

## Finalization (Sec. 3.2, Alg. 1 line 12, App. I)
- `"last_epoch_mean"` (default): record every point's assignment during the last epoch; centroid j =
  (weighted) mean of points assigned to j in that epoch. Guaranteed to decrease train MSE.
- `"slow_ema"`: extra (N^slow, S^slow) updated with the same assignments and the same splits, with a
  constant step `slow_ema_speed · ε · α0`; never used for assignment. Sec. 3.2 parametrises this by
  a half-life h; App. I / Table 15 by a speed multiplier → we use the speed form (Q-006).
- `None`: return the EMA centroids `S/N`.

## Auto hyperparameters (Eq. 6–7, Table 2)
- `B = 0.035 · k^0.13 · n^0.88 · T^0.81`; round to the nearest power of two; cap at n.
  For n > 2,000,000 use coefficient 0.0044 ("knee / 8").
- `α0 = min(0.824 · k^−0.46 · B^0.90 · n^−0.45 · T^−0.11, 1)` — computed with the RESOLVED B.
- Document in the docstring Notes: laws fitted only on DEEP (d = 96), 1e5 ≤ n ≤ 1e8,
  256 ≤ k ≤ 65,536, n/k ≥ 8 (Sec. 4.1, App. B.1); weaker on very low-dimensional / tiny data
  (Sec. 3.3 "Limitation", App. D.2).

## partial_fit
Streaming EMA with a CONSTANT α (no cosine, since there is no horizon M). Value: see Q-012 /
decision log. Document clearly that this is an engineering extension not evaluated in the paper.

## Explicitly out of scope
IVF assignment, low-precision assignment, multi-GPU sharding, any Faiss-only default
(d ≤ 16 quality batch law, revival off at d ≤ 2, B cap ≈ n/2) unless approved in decisionLog.md.
