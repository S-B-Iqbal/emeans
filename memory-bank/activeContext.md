# Active context

**Phase:** 9 - evidence guard, head-to-head, OpenMP numbers, example - **Mode:** ACT (approved)
**Status:** P9.1/P9.3/P9.4 done on `phase-9-evidence`; P9.2 is committed but its job can only be
dispatched once the workflow is on the default branch, i.e. after the merge. Gate pending.
`v0.1.0` is published as a **GitHub release** (D-060); the PyPI upload is blocked on credentials.
The #35383 comment **is posted** (D-061).

## Phase 9 results
- **P9.3** `benchmarks/tests/test_evidence.py` (6 tests) pins the headline relationships: e-means
  beats Lloyd@25 with revival, does **not** without it, one epoch suffices on the blob sets, the
  tuned MiniBatchKMeans is reported as ahead, and the stress-case loss stays reported.
- **P9.4** `docs/auto_examples/plot_emeans_embeddings.py`: 128-d embedding-like example comparing
  EMeans at four budgets with Lloyd@25 and MiniBatchKMeans; 1.8 s, deterministic, executed by the
  gallery under `sphinx -W -n`. At that size EMeans@3 is -35.4 % vs Lloyd@25 (0.04 s) but
  MiniBatchKMeans@3 is lower still - the example says so.
- **P9.1** `benchmarks/head2head.py` + `results/head2head.csv`: against `sklearn-contrib-emeans`
  0.3.2 at `OMP_NUM_THREADS=1`, both resolve `B = 8192`; ours -33.9/-15.5/-13.6 %, theirs
  -36.5/-14.7/-10.1 %; theirs 10-15 % faster. A reproducibility check, **not** a ranking (D-062).
  Written into `parity-matrix.md` and `benchmarks/report.md`.
- **P9.2** `.github/workflows/benchmark.yml` (manual dispatch, ubuntu, OpenMP, uploads
  `large-omp.csv`). `gh workflow run` 404s until the file is on `main`; dispatch after the merge
  and add the numbers to the report.

## Outstanding decisions
- **PyPI upload** needs credentials: `~/.pypirc`, or `TWINE_USERNAME=__token__` +
  `TWINE_PASSWORD`, or Trusted Publishing with a pending publisher. Command:
  `python -m twine upload -r pypi dist/*`. Nothing else is missing (artefacts verified).
- Enabling GitHub Pages for `docs-publish.yml` (a settings change).
- **Q-030** fractional `n_epochs`; posting the remaining drafts (Discussion P6.3, contrib P6.1,
  example PR P6.5) - each needs its own go-ahead.

## Phase history (merged)
- **P0-P5** scaffold (meson-python), NumPy reference, sklearn compliance (4 justified xfails),
  Cython/OpenMP kernels, honest benchmarks, packaging + docs + CI matrix.
- **P6** six upstream drafts in `docs/upstream/`; Linux wheels on `manylinux_2_28` (D-059).
- **P7** in-range benchmark (`n_train = 100 000`, `B = 8 192`, `alpha0 = 0.733`), user guide,
  docs-publish workflow, release prepared.
- **P8** `n_revivals_` + revival ablation (+2.2 %/+1.8 %/+0.8 % without it vs -33.9 %/-15.5 %/
  -13.6 % with it), anisotropic stress case, runner smoke tests, drafts refreshed.

## Notes for the next session
- Unit tests must stay fast: pass explicit `batch_size`/`n_epochs` rather than `"auto"`.
- New files under `emeans/` must be added to the matching `meson.build` `py.install_sources` list,
  then `pip install -e . --no-build-isolation`; `benchmarks/tests` needs its own pytest invocation.
- Repo-local git identity `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031);
  pushing `origin` branches is pre-authorised (D-030). Tags, `gh release`, PyPI upload, visibility
  changes, branch deletion and any write to another repository need explicit confirmation each time.
- Generated docs output (`docs/generated/`, `docs/sg_execution_times.rst`) is gitignored;
  `docs/auto_examples/` holds the sources and is tracked.
- `sklearn-contrib-emeans` 0.3.2 is installed in `.venv` for `benchmarks/head2head.py`; it does not
  clash with our `emeans` import name.
