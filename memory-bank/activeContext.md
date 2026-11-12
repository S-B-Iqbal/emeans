# Active context

**Phase:** 1 — Pure NumPy reference · **Mode:** ACT (user approved the plan in Act mode)
**Status:** Phase 0 approved and merged to `main`; P1.1 in progress on `phase-1-numpy-ref`.

## Phase 1 plan (approved)
Branch `phase-1-numpy-ref`, issue #2. New modules `emeans/_scaling_laws.py`, `_schedule.py`,
`_numpy_backend.py`, `_revival.py`, `_compat.py`, `_emeans.py` with matching tests in
`emeans/tests/`. Every new file must be listed in `emeans/meson.build` /
`emeans/tests/meson.build` — CI and `pytest --pyargs` only see the installed package.

- P1.1 `_scaling_laws.py`: Eq. 6–7 with Table 2 constants (3 s.f.), `round_to_power_of_two`
  (ties up, D-027), cap at `n`, coefficient 0.0044 above 2e6 (D-006), α0 from the **resolved** B.
- P1.2 `_schedule.py`: `n_updates = T·⌈n/B⌉` (D-004) and `alpha(t, M, α0, ε=0.1)` for `t = 1..M`
  (D-003), with `alpha(M) == ε·α0` exactly.
- P1.3 `_numpy_backend.py`: 256-row chunked expanded-form distances (D-028) with lowest-index ties,
  weighted `(n_j, s_j)` reduction in float64 (D-016), dense + CSR.
- P1.4 `_revival.py`: δ = 2⁻⁷ (D-010), dead = `N < 0.1·mean(N)`, donors ∝ `N` without replacement
  excluding dead/already-split clusters (D-013), `min(#dead, #donors)` per event (D-012), children
  from the donor centroid via the Eq. 8 pattern with `S = c·N` write-back; the slow state splits
  from its own centroid (D-035).
- P1.5 `EMeans.fit`: `N=1e-10`, `S=N·c⁰`, fresh permutation per epoch, global update counter
  (D-023), revival every `revival_period` (Alg. 1 line 9); `_compat.py` wraps the private sklearn
  imports; `_parameter_constraints` + `__sklearn_tags__`.
- P1.6 finalization: `last_epoch_mean` (recorded last-epoch assignments, empty-cluster fallback,
  D-009), `slow_ema` (constant `slow_ema_speed·ε·α0`, never used for assignment, D-008), `None`.
- P1.7 public API: `predict`, `transform` (Euclidean distances), `score`, `fit_predict`,
  `fit_transform`, `get_feature_names_out`, plus `labels_`/`inertia_` from one extra pass (D-018),
  `n_iter_`/`n_steps_` (D-019), resolved `batch_size_`/`alpha0_`.
- P1.8 `partial_fit`: constant step (D-014), first batch needs `n >= k` (D-036), `n_steps_` per call.
- P1.9 determinism, float32 preservation, `n < k` `ValueError`, out-of-range `UserWarning` (D-021).
- P1.10 numpydoc docstring with runnable Examples; `pytest --doctest-modules emeans` green.

Acceptance: every suite green locally **and** in CI; `pytest --cov=emeans --cov-fail-under=90`;
each non-obvious constant carries its paper citation; no Python-level loop over samples.

Risks: float-noise-sensitive ties (compare against a brute-force float64 reference), auto batch
size ≈ n on small test data (pass explicit `batch_size`), `last_epoch_mean` monotonicity only on
non-empty clusters (D-009), `_check_sample_weight` returning `X.dtype` weights (cast to float64).
Open questions touched: Q-027, Q-028 (both resolved as D-035/D-036).

## Phase 0 (done, approved)
meson-python skeleton, `emeans/__init__.py` + `py.typed` + smoke test, LICENSE/README/CITATION.cff,
minimal CI, decisions D-001…D-036, sklearn reference recorded in `systemPatterns.md`
(@7b2b14e0a89eedfa514dc9885e35a0ecade20cd0); repo private, issue #1 closed at the gate.

## Publishing (D-030…D-034)
- `origin` = `S-B-Iqbal/emeans` (private until the Phase 1 gate, then public — D-032); loop defined
  in `.clinerules/workflows/publish-progress.md`: push after every green task, tick the phase issue,
  merge to `main` only after a gate is approved.
- CI: `.github/workflows/tests.yml` (ubuntu, py3.10) on every push (D-034); full matrix in Phase 5.
  Green on `fix(ci): keep editable installs non-isolated…` (the very first run exposed the
  meson-python editable/build-isolation trap).
- Phase 0.6 (repo creation, LICENSE/README/CITATION, CI) is tracked as P0.6–P0.8 in `TASKS.md` and
  in issue #1.

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"` where
  the resolved batch size would be large.
- Repo-local git identity is `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031) and
  pushing to `origin` is pre-authorised (D-030); tags, releases, PyPI and other repositories are not.
