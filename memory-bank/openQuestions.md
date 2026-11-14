# Open questions
Format: Q-NNN · topic · paper reference · reference finding · proposed default · status

**Phase 0-5 questions are closed.** Every Phase 0 question was answered by the user and moved to
`memory-bank/decisionLog.md` (rule 02: approved questions move to the decision log and keep
their Q id there).

**OPEN (Phase 6, upstream work — need the user's answer before any code or draft changes):**

| Q | Topic | Context | Proposed default | Status |
|---|---|---|---|---|
| Q-030 | fractional `n_epochs` | The paper's Table-4 methodology uses 0.2-epoch steps; `sklearn-contrib-emeans` 0.3.1 accepts a fractional `n_epochs` ("the fractional pass comes before the last epoch, which is always a full pass"); we are integer-only by D-039 | keep integer for 0.1 (documented in the parity matrix); revisit only with explicit approval, since it changes the schedule contract | OPEN |
| Q-031 | package naming | Two e-means packages now exist: `sklearn-contrib-emeans` (import `sklearn_emeans`) and ours, `emeans-cluster` (import `emeans`). No import collision | keep `emeans-cluster`; do not rename | OPEN |
| Q-032 | offer `sample_weight` + `partial_fit` upstream | These are our two real API gaps relative to `sklearn-contrib-emeans` 0.3.1 (which has neither). Offering them publicly changes what this project is (option C in the Phase 6 plan) | keep them in our package only, and mention them in the thread as available for upstreaming *if asked* | OPEN |
| Q-033 | OpenMP wheels before publishing comparisons | Our benchmark numbers are serial (Apple M2, no `-fopenmp`); theirs ship OpenMP wheels, so wall-clock is not comparable. Options: label our numbers "serial", or vendor `libomp` for macOS wheels | label every number with its conditions in all Phase 6 drafts; no wheel-building change now | OPEN |

| Q | Decision | Topic |
|---|---|---|
| Q-001 | D-003 | cosine-schedule indexing (Alg. 1 `t = 1..M`) |
| Q-002 | D-004 | `M` when `B` does not divide `n` (keep the tail) |
| Q-003 | D-005 | batch-size cap at `n` |
| Q-004 | D-006 | batch-law switch at `n > 2e6` |
| Q-005 | D-007 | Table 2 coefficient precision |
| Q-006 | D-008 | slow-EMA parametrisation (speed form, start at `t = 0`) |
| Q-007 | D-009 | revival during the last epoch |
| Q-008 | D-010 | `delta = 2**-7` for every dtype |
| Q-009 | D-011 | zero coordinates in Eq. 8 |
| Q-010 | D-012 | more dead clusters than eligible donors |
| Q-011 | D-013 | donor pool |
| Q-012 | D-014 | `partial_fit` constant step |
| Q-013 | D-015 | k-means++ variant (greedy) |
| Q-014 | D-016 | float64 accumulation |
| Q-015 | D-017 | `n` with `sample_weight` |
| Q-016 | D-018 | `labels_` / `inertia_` extra pass |
| Q-017 | D-019 | `n_iter_` vs `n_steps_` |
| Q-018 | D-020 | CSR support |
| Q-019 | D-021 | tiny `n`, `n/k < 8`, low `d` |
| Q-020 | D-022 | `N = 1e-10` in float64 |
| Q-021 | D-023 | global revival counter |
| Q-022 | D-024 | sample_weight test design |
| Q-023 | D-025 | Faiss-only defaults not adopted |
| Q-024 | D-026 | `init="random"` semantics with weights |
| Q-025 | D-027 | power-of-two rounding rule |
| Q-026 | D-028 | assignment arithmetic and tie-breaking |
| Q-027 | D-035 | slow-EMA split geometry (children from the slow donor centroid) |
| Q-028 | D-036 | first `partial_fit` batch needs `n >= k` |
| Q-029 | D-052 | PyPI name `emeans` taken → distribution `emeans-cluster` |

New ambiguities are appended here as `Q-NNN` with status `OPEN` via `/open-question.md`, and
move to the decision log only after user approval.
