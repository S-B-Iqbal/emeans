# Active context


**Phase:** 7 - scale-up benchmark, release prep, docs guide - **Mode:** ACT (approved by the user)
**Status:** P7.0-P7.3 done on `phase-7-scaleup`; gate pending. Branch is 5 commits ahead of `main`.
P7.4 (Q-030 fractional `n_epochs`) and P7.5 (Q-033 macOS OpenMP wheels) are parked pending decisions.

## Phase 7 results
- **P7.0/P7.3** `docs/guide.rst` (new user guide, in the toctree), `docs-publish.yml` (manual
  trigger: enabling Pages is a settings change and stays gated), and the false install claim in
  `docs/upstream/sklearn-35383-comment.md` is fixed.
- **P7.1** `benchmarks/run_suite.py large`: `n_train = 100 000`, `k = 512`, `d = 32/96`,
  `B = 8 192`, `alpha0 = 0.733`, MiniBatchKMeans swept at **every** epoch budget.
  Results (median of 5 seeds, test MSE vs Lloyd@25): epochs-to-reference **1** (9/10 seeds;
  paper 1.4-2.6), d=96 EMeans@1 -19.0%, @5 **-33.9%** (1.5 s) vs Lloyd@25 2.3 s, tuned
  MiniBatch@5 **-47.3%** (14 s); d=32 EMeans@5 -15.5% vs MiniBatch@5 -41.1%. Honest verdict in
  `benchmarks/report.md`: e-means wins on epochs and on time at d=96, loses to the tuned
  MiniBatchKMeans at equal epochs in both datasets and at equal time at d=32.
- **P7.2** release prepared: version `0.1.0` in `meson.build` + CHANGELOG dated, `python -m build`
  produces `emeans_cluster-0.1.0{,.tar.gz}`, `twine check` PASSED, clean-venv install reports
  `0.1.0` with `has_openmp() == False` locally (macOS).  Gated and NOT executed: `git tag`,
  `twine upload`, `gh release create`.
- Traps recorded in `docs/release-runbook.md` and `progress.md`: `meson dist` archives git HEAD
  (commit the version bump first), and `ruff format` reformats Python inside Markdown.

## Outstanding decisions (asked, not answered)
- **Q-032**: offer `sample_weight` + `partial_fit` upstream? Needed to freeze the #35383 text.
- Enabling GitHub Pages (settings change) to make `docs-publish.yml` useful.
- Release go-ahead for the three gated commands above.
- **Q-030** fractional `n_epochs` (P7.4), **Q-033** OpenMP macOS wheels (P7.5).
- Posting any `docs/upstream/` draft still needs a separate explicit go-ahead (rule 01).

## Phase 6 (closed)
Merged to `main` as `d7786c8`; issue #7 closed; six drafts in `docs/upstream/`, nothing posted.
Linux wheels build on `manylinux_2_28` (D-059); wheel self-test `openmp: True` on Linux, `False` on
macOS/Windows.

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"`.
- Repo-local git identity `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031);
  pushing `origin` branches is pre-authorised (D-030). Tags, `gh release`, PyPI upload, visibility
  changes, branch deletion and any write to another repository need explicit confirmation each time.
- Generated docs output (`docs/generated/`, `docs/sg_execution_times.rst`) is gitignored;
  `docs/auto_examples/` holds the sources and is tracked.
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
