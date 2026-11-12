# Open questions
Format: Q-NNN · topic · paper reference · reference finding · proposed default · status

Q-001 · t indexing · Eq. 5 says t ∈ {0..M−1}; Alg. 1 line 6 does `t ← t+1` before γ(t) ·
  — · Follow Eq. 5: first update uses α(0)=α0, last uses α(M−1). Revival after batches 5,10,…
  (1-based count, matching Alg. 1 line 10). · OPEN
Q-002 · M when B ∤ n · Sec. 3.1: M = T·⌊n/B⌋ · Faiss README: last batch of each pass is the
  leftover · Keep the leftover as a smaller batch; schedule horizon M = T·⌈n/B⌉ so the cosine
  ends at its floor; n_steps_ = M. · OPEN
Q-003 · B cap · Sec. 3.3: capped at n · Faiss: ≈ n/2 · Cap at n (paper). Log divergence. · OPEN
Q-004 · Batch-law switch · Sec. 3.3: n > 2M → ÷8 · Faiss "auto": quality law at d ≤ 16 ·
  Paper rule only; mention Faiss behaviour in Notes. · OPEN
Q-005 · Coefficient precision · Table 2 (3 sig. figs) · Faiss README uses 6 sig. figs ·
  Use Table 2 values. · OPEN
Q-006 · Slow-EMA parametrisation · Sec. 3.2 half-life h vs App. I / Table 15 speed multiplier ·
  — · Speed form: constant step = slow_ema_speed·ε·α0; slow state initialised as a copy of the
  fast state at the start of the last epoch (alternative: from t=0); receives same splits. · OPEN
Q-007 · Revival during last epoch · not specified · — · Keep revival active; last-epoch labels
  remain valid (pre-split points stay with the donor index); clusters with no last-epoch points
  keep their EMA centroid. · OPEN
Q-008 · δ for float32/float64 · App. H.1: BF16 machine eps · — · δ = 2⁻⁷ for all dtypes. · OPEN
Q-009 · Zero coordinates / zero-vector donor · Eq. 8 multiplicative split · — · Follow paper
  (children coincide on zero coords; ties → lower index). Optional additive guard OFF by default. · OPEN
Q-010 · #dead > #eligible donors · App. H.1 · — · Revive min(#dead, #live) per event, lowest-N
  first; remainder waits for next period. · OPEN
Q-011 · Donor pool · implicit · — · Exclude dead clusters and clusters already split in this event. · OPEN
Q-012 · partial_fit α · not in paper · Faiss has no partial_fit · If alpha0 is a float use it;
  if "auto": α = ε·α0_law(k, B=len(batch), n=n_seen, T=1), stored in alpha0_.
  ENGINEERING CHOICE — needs approval. · OPEN
Q-013 · k-means++ variant · App. G.2 greedy (sklearn) better than original · — ·
  Use `sklearn.cluster.kmeans_plusplus` (greedy). · OPEN
Q-014 · float32 accumulation · App. J keeps counts/sums/EMA in FP32 even for BF16 assignment ·
  — · Accumulate batch reductions in float64; store N,S in float64; cast centers to X dtype.
  Revisit after a drift test. · OPEN
Q-015 · n in the laws with sample_weight · not specified · — · n = number of rows (not Σw). · OPEN
Q-016 · labels_/inertia_ · Alg. 1 returns centroids only · Faiss returns last-epoch labels ·
  One extra assignment pass with final centroids so labels_ == predict(X) (as MiniBatchKMeans
  compute_labels=True). Benchmarks report with/without this pass. · OPEN
Q-017 · n_iter_ vs n_steps_ · — · — · n_iter_ = epochs completed; n_steps_ = EMA updates. · OPEN
Q-018 · Sparse CSR · not covered · Faiss: dense only · S, centers dense (k,d); distances via
  ‖x‖² − 2x·c + ‖c‖² with CSR dot products. · OPEN
Q-019 · n < k / n/k < 8 / tiny n · laws fitted on n/k ≥ 8, 1e5 ≤ n ≤ 1e8 · Faiss: revival off at
  d ≤ 2 · ValueError if n < k; UserWarning if auto laws are used outside the fitted range. · OPEN
Q-020 · Initialization N_j=1e-10 with float32 S · Sec. 3.1 · — · N,S stored float64 (see Q-014). · OPEN
Q-021 · Revival period counted in batches across epochs (not reset per epoch) · Alg. 1 uses
  global t · — · Global counter. · OPEN
Q-022 · sample_weight equivalence test design · — · — · Exact in deterministic regime
  (B = n, α constant = 1, finalization=None ⇒ Lloyd step); statistical otherwise. · OPEN
Q-023 · Faiss-only defaults (d ≤ 16 law, revival off at d ≤ 2) · not in paper · Faiss PR #5854 ·
  Not adopted in v0.1; possible opt-in later (new hyperparameter → needs approval). · OPEN
