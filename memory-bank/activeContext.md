# Active context


**Phase:** 6 - Upstream drafts - **Mode:** ACT (approved with Phase 5, option B)
**Status:** P6.0-P6.5 done on `phase-6-upstream`; gate pending. Six drafts in `docs/upstream/`,
nothing posted (rule 01/10). Also fixed the Linux wheel build (manylinux_2_28).

## Phase 6 (done, gate pending)
- P6.0 parity matrix: paper | `emeans-cluster` | `sklearn-contrib-emeans` 0.3.1 | Faiss.
- P6.1 contrib request (requirements quoted from `workflow.md`; four gaps stated; the
  duplicate-project question and the `sklearn-contrib-*` name collision are put to the org).
- P6.2 comment for scikit-learn #35383 (≤350 words; serial-CPU numbers incl. the losses).
- P6.3 MiniBatchKMeans improvement discussion (four landable items, written against the real
  code in `_ref/scikit-learn`: count-based running mean, `reassignment_ratio`, `_counts`).
- P6.4 PR sketch (API, files, kernel change, tests, back-compat).
- P6.5 user-example skeleton for embedding clustering (`betatim`'s request).
- Open questions filed: Q-030 fractional `n_epochs`, Q-031 naming, Q-032 offer
  `sample_weight`/`partial_fit` upstream, Q-033 OpenMP wheels (all with proposed defaults).
- Upstream facts (2026-11-14): #35383 OPEN, 25 comments, `Needs Decision - Include Feature`;
  `sklearn-contrib-emeans` 0.3.1 released 13 Nov 2026; agreed roadmap = package → contrib → 1.10.

## Phases 0-5 (done, merged)
- P5 packaging: `emeans-cluster` (D-052), docs (`sphinx -W -n`), CI 3 OS x 3.10-3.13 + docs +
  coverage + nightly + cibuildwheel; `main` = Phase 5 merge, CI green (14/14 jobs).
- P1-P4: NumPy reference, sklearn compliance (4 justified xfails), Cython/OpenMP kernels,
  honest benchmarks (loses to a tuned MiniBatchKMeans on 3/4 laptop configs).

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"`.
- Repo-local git identity `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031);
  pushing `origin` branches is pre-authorised (D-030). Tags, `gh release`, PyPI upload,
  visibility changes and any write to another repository need explicit confirmation each time
  — including posting any of the Phase 6 drafts.
- Generated docs output (`docs/generated/`, `docs/sg_execution_times.rst`) is gitignored;
  `docs/auto_examples/` holds the sources and is tracked.
