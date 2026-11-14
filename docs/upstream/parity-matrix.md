# Parity matrix — e-means implementations (P6.0, draft)

Facts only, each cell traceable to a source. `n/d` = not documented in the source we read.
`D-NNN` = `memory-bank/decisionLog.md`.

Sources

| # | Source | How we used it |
|---|---|---|
| 1 | Paper: Bessa et al., *e-means*, arXiv:2611.09501v1 (`docs/PAPER_NOTES.md`) | normative |
| 2 | `emeans-cluster` 0.1.0.dev0 — this repository (`emeans/_emeans.py`, `decisionLog.md`) | our implementation |
| 3 | `sklearn-contrib-emeans` **0.3.1** — PyPI project description, fetched 2026-11-14 (import name `sklearn_emeans`, repo `cakedev0/sklearn-contrib-emeans`) | independent implementation, read only |
| 4 | Faiss `contrib/e_means` (Faiss 1.16.0; `docs/REFERENCES.md`) | behavioural cross-check only, never copied |

## Core algorithm

| Behaviour | Paper | `emeans-cluster` | `sklearn-contrib-emeans` 0.3.1 | Faiss contrib |
|---|---|---|---|---|
| State | counts `N_j` + sums `S_j`, `c_j = S_j/N_j` (Eq. 4), centroids never stored | same (float64 state, D-016) | "each center is an EMA of the samples assigned to it" | same |
| Update rule | `N ← γN + (1−γ)n`, `S ← γS + (1−γ)s` (Eq. 2/3) | same, weighted by `sample_weight` | same | same |
| Step schedule | cosine from α0 down to `ε·α0`, ε = 0.1 (Eq. 5) | same (`alpha_floor_ratio` exposed, default 0.1) | "cosine schedule on the EMA step" | cosine |
| Schedule indexing | `t ∈ {0…M−1}` (Eq. 5) but Alg. 1 increments `t` before use | `t = 1…M` (D-003; last step lands exactly on `ε·α0`) | n/d | n/d |
| Updates per epoch | `M = T·⌊n/B⌋` | `T·⌈n/B⌉`, tail batch kept (D-004, same as Faiss) | n/d | tail batch kept |
| Batch size | law of Table 2, rounded to a power of two, capped at `n` (Eq. 7) | same, cap at `n` (D-005), `/8` coefficient above 2e6 (D-006) | "comes from scaling laws fitted by the authors" | three laws (`throughput`, `quality`, `auto`), cap ≈ `n/2` |
| Peak step α0 | law of Table 2, `min(…, 1)` (Eq. 6) | same, computed from the resolved `B` (D-006) | same origin | same law |
| Extrapolation | laws fitted on DEEP (d = 96), `1e5 ≤ n ≤ 1e8`, `n/k ≥ 8` | `UserWarning` outside that range (D-021); no correction | n/d | `quality` law at `d ≤ 16` (PR #5854) |
| Dead-cluster revival | every 5 batches, `N_j < τ·mean(N)`, τ = 0.1; donor ∝ `N` without replacement; Eq. 8 split | same; `revival_threshold`, `revival_period` exposed; always on (τ is the only knob) | not mentioned in the description | "off" at `d ≤ 2` (PR #5854) |
| Split geometry | `c ⊕ [1±δ, 1∓δ, …]`, δ = BF16 ε = 2⁻⁷ | same, `δ = 2**-7` for every dtype (D-010) | n/d | same procedure |
| Finalization | last-epoch mean (default) or slow EMA (Sec. 3.2) | `last_epoch_mean` \| `slow_ema` \| `None`, plus `slow_ema_speed` (D-008) | n/d beyond the EMA centers | last-epoch mean |
| `labels_` / `inertia_` | not specified | extra assignment pass, so `labels_ == predict(X)` (D-018) | "extra assignment pass on the final centers" | from the last training epoch |
| Epochs | `T` integer; protocol explores 0.2-epoch steps (Table 4) | integer `T ≥ 1` only (D-039) | **fractional** — "the fractional pass comes before the last epoch" | integer |

## API surface

| Behaviour | Paper | `emeans-cluster` | `sklearn-contrib-emeans` 0.3.1 | Faiss contrib |
|---|---|---|---|---|
| Signature | — | `n_clusters, *, n_epochs=5, batch_size="auto", alpha0="auto", alpha_floor_ratio=0.1, init="random", revival_threshold=0.1, revival_period=5, finalization="last_epoch_mean", slow_ema_speed=1.0, random_state=None, verbose=0` | `n_clusters=8, *, init="random", n_epochs=5, random_state=None` (+ array-API dispatch) | `k`, epochs, batch size, step |
| `init` | random (uniform, no replacement) | `"random"`, `"k-means++"`, array (D-015) | `"random"`, `"k-means||"`, `"k-means++"`, array, callable | random |
| `sample_weight` | weighted form not given (w ≡ 1) | **supported** in `fit` and `partial_fit`, float64 accumulation (D-016, D-024) | **not supported** ("There is no sample_weight yet") | no |
| `partial_fit` | not specified | **supported**, constant step (D-014), documented as an engineering extension | not mentioned in the description | no |
| Sparse input | not specified | CSR in and out (D-020) | CSR accepted, other formats converted | dense float32 only |
| dtype | not specified | float32 preserved, float64 state | float32 (faster) or float64 | dense float32 only |
| Public API | — | `fit`, `partial_fit`, `predict`, `transform`, `score`, `fit_predict`, `fit_transform`, `get_feature_names_out` | `fit`, `predict`, `fit_predict`, `transform`, `fit_transform`, `score` | train / assign |
| Fitted attributes | — | `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, `n_steps_`, `n_features_in_`, `feature_names_in_`, `batch_size_`, `alpha0_` | `cluster_centers_`, `labels_`, `inertia_`, `batch_size_` | centroids |
| Threads | 64-core Xeon + Faiss assignment (Sec. 4.1) | OpenMP on Linux, serial on macOS (D-044) and Windows (D-056); `has_openmp()` reports it | OpenMP wheels, "uses all the physical cores or `OMP_NUM_THREADS`" | Faiss OpenMP |
| Array API / GPU | out of scope for this project (App. J/M) | not supported | PyTorch/CuPy via `array_api_dispatch=True` | GPU through Faiss |
| Minimum scikit-learn | — | 1.6 (`validate_data`, `__sklearn_tags__`) | 1.7 | n/a |
| Compliance | — | `parametrize_with_checks` green, 4 justified xfails (D-041); 269 tests; coverage gate 90 % | runs the scikit-learn estimator checks (per project description) | n/a |
| Distribution | — | `emeans-cluster` on PyPI (import `emeans`, D-052); **no wheels published yet** | `sklearn-contrib-emeans` with wheels on PyPI since 13 Nov 2026 | inside Faiss |
| Licence | — | BSD-3-Clause, independent implementation, no Faiss code | not checked here | MIT |

## What each project has that the other does not

`emeans-cluster` (ours)

- `sample_weight` in `fit` and `partial_fit`, with an exact equivalence test in the deterministic
  regime (D-024).
- Streaming `partial_fit`.
- Three finalization modes including slow EMA plus its speed knob (Sec. 3.2 / App. I).
- Exposed `alpha_floor_ratio`, `revival_threshold`, `revival_period` for ablations; the revival
  invariants (ΣN conserved, `S = c·N` after a split, donor ∝ N) are unit-tested.
- A published, reproducible benchmark report **including its losing cases**
  (`benchmarks/report.md`).

`sklearn-contrib-emeans` (theirs)

- Published OpenMP wheels for Linux/macOS since 13 Nov 2026, with PyPI attestations.
- Fractional `n_epochs` (the paper's 0.2-epoch protocol), `k-means||` init, callable init.
- Array API support (PyTorch/CuPy) and a longer history toward contrib inclusion.

## Caveats that must accompany any posted number

1. Our wall-clock figures come from an **Apple M2 with serial kernels**; theirs come from OpenMP
   wheels. The two are not comparable (Q-033): compare *epochs*, never seconds.
2. Our measurements are at `n_train = 10 000` with `k = 512`; the laws are fitted for
   `1e5 ≤ n ≤ 1e8`. Below that range we warn (D-021) and the numbers are extrapolations.
3. The Faiss column is second-hand: behavioural observations recorded in `docs/REFERENCES.md`
   from the released `contrib/e_means`; no Faiss code was read or copied (rule 05).
