# Active context

**Phase:** 0 — Plan · **Mode:** ACT (executing the approved Phase 0 plan)
**Status:** P0.1–P0.5 done and green; gate summary posted, waiting for the user to say
`approved phase 0`.

## What exists now
- meson-python skeleton: `pyproject.toml`, `meson.build` (`find_installation(pure: false)`,
  dynamic version from `project(version:)`), `emeans/meson.build`, `emeans/tests/meson.build`.
- Package: `emeans/__init__.py` (version + doctest only — `EMeans` arrives in P1.5),
  `emeans/py.typed`, `emeans/tests/test_package.py` smoke test.
- `pip install -e . --no-build-isolation` works; version `0.1.0.dev0`.
- Decisions D-001…D-029 recorded; no open questions left (`openQuestions.md` is a pointer table).
- scikit-learn reference studied; patterns written into `systemPatterns.md`
  (@7b2b14e0a89eedfa514dc9885e35a0ecade20cd0).

## Next steps (only after `approved phase 0`)
1. `/new-phase.md` → plan Phase 1 here, then `git checkout -b phase-1-numpy-ref`.
2. Phase 1 order: P1.1 scaling laws → P1.2 schedule → P1.3 NumPy backend → P1.4 revival →
   P1.5 `fit` loop → P1.6 finalization → P1.7 public methods → P1.8 `partial_fit` →
   P1.9 determinism/dtype → P1.10 docstring doctests.
3. Every constant keeps its paper citation (`Eq. N`, `Table N`, `Alg. 1 line N`, `App. X.Y`).

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"` where
  the resolved batch size would be large.
- Repo-local git identity is the placeholder `Cline (placeholder) <cline@localhost>` (D-029);
  the user should set their own before Phase 5 publishes anything.
