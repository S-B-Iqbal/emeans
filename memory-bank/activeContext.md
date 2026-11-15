# Active context

**Phase:** 8 - evidence hardening - **Mode:** ACT (approved, merged `af69786`)
**Status:** Phases 0-8 are **closed and merged to `main`**. No phase is running: the next one needs
explicit approval (proposal below). Nothing has been posted upstream, tagged, uploaded or released.

## Proposed Phase 9 (ranked; not started)
- **P9.1** head-to-head against `sklearn-contrib-emeans` **0.3.2**: isolated venv,
  `OMP_NUM_THREADS=1` for both, our three in-range data sets with `n_epochs` matched; record test
  MSE and wall-clock and replace the `n/d` cells in `docs/upstream/parity-matrix.md` with measured
  values. The one question our own evidence cannot answer.
- **P9.4** embedding-clustering example in *our* gallery
  (`docs/auto_examples/plot_emeans_embeddings.py`, from the P6.5 sketch) - the example `betatim`
  asked for, done where no permission is needed.
- **P9.2** post the #35383 comment (gated, per draft; needs Q-032 answered).
- **P9.3** execute the 0.1.0 release and enable Pages (gated, per command).
- Parked until answered: Q-030 fractional `n_epochs` (P9.5), Q-033 macOS OpenMP wheels (P9.6),
  k-means|| (P9.7).

## Upstream state (2026-11-14, re-checked)
- scikit-learn #35383: OPEN, unchanged since 13 Nov; `betatim`'s request for an embedding-clustering
  example is still unanswered, and `glevv` suggested `MiniBatchKMeans(algorithm="e-means")`.
- `sklearn-contrib-emeans`: **0.3.2 released 2026-11-14 20:26** (six releases in three days).
  Has OpenMP wheels, CSR, fractional `n_epochs`, k-means||, array API; no `sample_weight`,
  no `partial_fit`. Still **no** scikit-learn-contrib inclusion request filed (the slot is open, but
  inclusion would require a repo transfer and the `sklearn-contrib-*` name, which is taken).

## Phase history (all merged)
- **P8** `n_revivals_` + revival ablation (without revival: +2.2 %/+1.8 %/+0.8 % vs -33.9 %/-15.5 %/
  -13.6 % with it), anisotropic stress case, 7 benchmark-runner smoke tests in CI, drafts refreshed
  with in-range numbers (comment 348 words).
- **P7** in-range benchmark (`n_train = 100 000`, `B = 8 192`, `alpha0 = 0.733`, one epoch to
  Lloyd@25 in 14/15 runs), user guide, docs-publish workflow, release prepared (build + `twine
  check` PASSED + clean-venv install at `0.1.0`).
- **P6** six upstream drafts in `docs/upstream/`, none posted; Linux wheels on `manylinux_2_28`
  (D-059).
- **P0-P5** scaffold (meson-python), NumPy reference, sklearn compliance (4 justified xfails),
  Cython/OpenMP kernels, honest benchmarks, packaging + docs + CI matrix.

## Outstanding decisions (asked, not answered)
- **Q-032** offer `sample_weight` + `partial_fit` upstream? Needed before the #35383 comment goes out.
- Release go-ahead: `git tag -a v0.1.0`, `python -m twine upload -r pypi dist/*`,
  `gh release create v0.1.0 dist/*`, plus enabling GitHub Pages for `docs-publish.yml`.
- **Q-030** fractional `n_epochs`, **Q-033** OpenMP macOS wheels; posting any draft (rule 01).

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"`.
- New files under `emeans/` must be added to the matching `meson.build` `py.install_sources` list,
  then `pip install -e . --no-build-isolation`; `benchmarks/tests` needs its own explicit pytest
  invocation (`--pyargs emeans` cannot see it).
- Repo-local git identity `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031);
  pushing `origin` branches is pre-authorised (D-030). Tags, `gh release`, PyPI upload, visibility
  changes, branch deletion and any write to another repository need explicit confirmation each time
  - including posting any `docs/upstream/` draft.
- Generated docs output (`docs/generated/`, `docs/sg_execution_times.rst`) is gitignored;
  `docs/auto_examples/` holds the sources and is tracked.
