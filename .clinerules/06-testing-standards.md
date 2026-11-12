---
paths:
  - "emeans/**"
  - "benchmarks/**"
---
# 06 — Testing standards

## Required suites (Phase 1 unless noted)
| File | Covers |
|---|---|
| test_scaling_laws.py | Table 2 formulas exactly; power-of-two rounding; cap at n; n > 2M switch; α0 ≤ 1 |
| test_schedule.py | Eq. 5: α(0) = α0, α(M−1) ≈ ε·α0 (+cos term), monotone non-increasing |
| test_revival.py | Σ N conserved; S = c·N after split; Eq. 8 sign pattern; donor ∝ N (statistical); no donor reuse; dead excluded from donors |
| test_finalization.py | last_epoch_mean train MSE ≤ EMA centroids; slow_ema not used for assignment; None returns S/N |
| test_emeans.py | fit/predict/transform/score shapes; determinism with random_state; float32 preservation; n < k error; verbose |
| test_partial_fit.py | constant α; n_steps_ increments; reset=False feature checks |
| test_common.py (P2) | `parametrize_with_checks([EMeans(...)])` |
| test_sparse.py (P2) | CSR results == dense results (same seed) |
| test_sample_weight.py (P2) | integer weights == repeated rows in deterministic regime (B = n, constant α = 1); statistical test otherwise |
| test_backends.py (P3) | NumPy == Cython: identical labels (ties documented), allclose sums/counts |

## Rules
- Parametrize stochastic tests over several seeds; keep tests < 2 s each on a laptop (small n, k, d).
- Use `numpy.testing.assert_allclose` with explicit `rtol` per dtype (float32: 1e-5, float64: 1e-12).
- Any `expected_failed_checks` entry needs a justification recorded in decisionLog.md.
- Doctests run: `pytest --doctest-modules emeans`.
- Coverage ≥ 90% lines on `emeans/` (excluding `_kernels` generated C).
- Never weaken an assertion to pass; never delete a failing test without approval.
