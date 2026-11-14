# Active context


**Phase:** 6 - Upstream drafts - **Mode:** ACT (approved, merged `d7786c8`)
**Status:** Phase 6 **closed** (issue #7 closed, `main` = `d7786c8`). Nothing posted upstream.
No phase is running; the next one needs explicit approval. Outstanding decisions are listed below.

## Blocking decisions before any upstream post
1. **Known defect**: `docs/upstream/sklearn-35383-comment.md` opens with `pip install
   emeans-cluster`, which does not exist on PyPI yet. Either drop the clause or release first
   (P7.2). Must be fixed before posting.
2. **Q-032**: offer `sample_weight` + `partial_fit` upstream (changes the last paragraph of the
   #35383 comment and item 3 of the Discussion).
3. **Posting order approved**: #35383 comment (P6.2) first, then the embedding example (P6.5) in
   the same thread; hold the Discussion (P6.3) and the contrib request (P6.1) until a maintainer
   signals the route or we have a release.
4. **Q-030** fractional `n_epochs`, **Q-033** OpenMP macOS wheels — decide only if Phase 7 runs.

## Proposed Phase 7 (ranked; not started)
- **P7.1** benchmark inside the fitted range: `n = 100 000`, `d = 96`, `k = 512`. Our resolver gives
  `B = 8 192`, `alpha0 = 0.733`, 13 batches/epoch — the regime the paper claims, which retires the
  "below the fitted range" caveat carried by every draft.
- **P7.2** release `emeans-cluster` 0.1.0 (runbook exists; tag/twine/`gh release` stay gated).
- **P7.3** user guide page + `gh-pages` (the two remaining contrib gaps).
- **P7.4** Q-030 fractional `n_epochs`; **P7.5** Q-033 macOS OpenMP wheels (libomp).
- Lowest priority: k-means|| init.

## Upstream state (2026-11-14, re-checked)
- scikit-learn #35383 OPEN, quiet since 13 Nov; `betatim` asked for an embedding-clustering
  example, which is exactly our winning regime (d = 96: -19.8% vs Lloyd@25 at 5 epochs).
- scikit-learn-contrib: **no e-means inclusion request filed yet**; `workflow.md` requires the
  PyPI name `sklearn-contrib-<project>` (taken) and includes a repo transfer on acceptance.
- `sklearn-contrib-emeans`: 0.2.0 (12 Nov) and 0.3.1 (13 Nov); OpenMP wheels, CSR, fractional
  epochs, k-means||, array API; no `sample_weight`, no `partial_fit`.

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"`.
- Repo-local git identity `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031);
  pushing `origin` branches is pre-authorised (D-030). Tags, `gh release`, PyPI upload,
  visibility changes, branch deletion and any write to another repository need explicit
  confirmation each time — including posting any Phase 6 draft.
- Generated docs output (`docs/generated/`, `docs/sg_execution_times.rst`) is gitignored;
  `docs/auto_examples/` holds the sources and is tracked.
