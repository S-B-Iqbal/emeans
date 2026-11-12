# 00 — Project overview: `emeans`

## Mission
Build `EMeans`, a production-quality, scikit-learn-compatible implementation of **e-means**
(Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou — "e-means: outperforming k-means
(in one-tenth the epochs)", arXiv:2611.09501v1, FAIR at Meta / École des Ponts, 6 Nov 2026),
shipped as the standalone package `emeans`.

## Targets
1. GitHub + PyPI release (BSD-3-Clause).
2. Proposal to scikit-learn-contrib.
3. Upstream conversation with scikit-learn — **NOT** a new-estimator PR (FAQ inclusion bar:
   ≥3 years since publication, ≥200 citations). Join existing issue #35383
   ("Add e-means clustering (EMeans)") and pitch `MiniBatchKMeans` improvements.

## Source of truth (in priority order)
1. The paper: `docs/e-means.pdf` (HTML mirror: https://arxiv.org/html/2611.09501v1).
   Digest: `docs/PAPER_NOTES.md`. Cite `Eq. N`, `Table N`, `Alg. 1 line N`, `App. X.Y` in comments.
2. `memory-bank/decisionLog.md` — approved resolutions of ambiguities.
3. scikit-learn source in `_ref/scikit-learn` (READ-ONLY) for structure, naming, validation style.
4. Faiss contrib / website / X post — behavioural cross-checks only (see `docs/REFERENCES.md`).

## Non-goals for v0.x
GPU kernels, BF16/FP8/INT8 assignment (App. J), IVF approximate assignment (Sec. 3.4, App. M),
multi-node sharding (App. M.1).

## Mode discipline
- Start every phase and every non-trivial task in **PLAN** mode.
- Write the plan into `memory-bank/activeContext.md`; switch to **ACT** only after the user approves.
