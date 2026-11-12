# Open questions
Format: Q-NNN · topic · paper reference · reference finding · proposed default · status

**No open questions.** Every Phase 0 question was answered by the user and moved to
`memory-bank/decisionLog.md` (rule 02: approved questions move to the decision log and keep
their Q id there).

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

New ambiguities are appended here as `Q-NNN` with status `OPEN` via `/open-question.md`, and
move to the decision log only after user approval.
