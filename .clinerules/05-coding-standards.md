# 05 — Coding standards

## Provenance (non-negotiable)
- Implement from the paper. Do NOT open, copy, port, or paraphrase code from
  `facebookresearch/faiss/contrib/e_means`. Its README/PR text may be read to cross-check behaviour;
  record findings in prose in `openQuestions.md` as "reference agrees / differs".
- Every non-obvious constant carries a citation comment: `# Table 2`, `# App. H.1`, `# Eq. 5`.

## Python
- Python ≥ 3.10. Type hints on all private helpers; public API typed in docstrings (numpydoc).
- ruff-check + ruff-format (black-compatible), line length 88.
- Pure functions for math (`_scaling_laws.py`, `_schedule.py`, `_revival.py`) so they are unit-testable
  without the estimator.
- No Python-level loops over samples in hot paths; loops over clusters only where k is small or in Cython.
- File header: `# SPDX-License-Identifier: BSD-3-Clause`.

## Numerics
- Preserve the input dtype (float32 stays float32) for X, centers, transform output.
- Reductions over a batch accumulate in float64 when X is float32 unless Q-014 decides otherwise.
- Never divide by N without the 1e-10 floor guaranteed by initialization (Eq. 4); assert finite in debug.
- Use `np.random.RandomState` from `check_random_state`; no global RNG, no `np.random.seed`.

## Structure
```
emeans/_emeans.py         EMeans estimator only
emeans/_scaling_laws.py   Eq. 6–7, Table 2
emeans/_schedule.py       Eq. 5
emeans/_revival.py        App. H.1
emeans/_numpy_backend.py  reference assignment + count/sum reduction
emeans/_kernels/*.pyx     Phase 3 only
emeans/_compat.py         wrappers for private sklearn imports
```
A private `_backend` switch ("numpy" | "cython") selects the kernel; NumPy stays as fallback forever.

## Logging
`verbose > 0` prints per-epoch: epoch, steps, current α, batch inertia, #revived. No `print` otherwise.
