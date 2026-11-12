# Tech context

- Python 3.10–3.13 · numpy · scipy · scikit-learn>=1.6 · threadpoolctl
- Build: meson-python + meson + ninja + Cython>=3 (Phase 3 adds extensions); flat layout.
- Dev install: `pip install -e . --no-build-isolation` with the venv activated (meson-python
  shells out to the `meson` executable, so `.venv/bin` must be on `PATH`)
- Tests: `pytest --pyargs emeans`, `pytest --doctest-modules emeans`
- Lint: pre-commit → ruff-check, ruff-format, cython-lint, basic hygiene hooks
- Docs: sphinx, numpydoc, sphinx-gallery, pydata-sphinx-theme
- CI (Phase 5): GitHub Actions matrix ubuntu/macos/windows × 3.10–3.13; wheels via cibuildwheel;
  macOS wheels vendor libomp; nightly-sklearn job.
- Read-only reference: `_ref/scikit-learn` @ `7b2b14e0a89eedfa514dc9885e35a0ecade20cd0`
  (main branch requires Python >= 3.12: structure reference only; the behavioural reference is
  the installed scikit-learn 1.7.2).
- `docs/e-means.pdf` (2.18 MB) is gitignored — it exceeds the `check-added-large-files
  --maxkb=1024` hook; `scripts/bootstrap.sh` re-downloads it.

## Publishing (D-030..D-034)
- Repository: `S-B-Iqbal/emeans`, `origin` = <https://github.com/S-B-Iqbal/emeans> — **private**
  until the Phase 1 gate, then public (D-032).
- `gh` 2.100.0 authenticated as `S-B-Iqbal`; token scopes `gist, read:org, repo, workflow`
  (no `project`, so no GitHub Projects; tracking uses issues, D-033).
- CI: `.github/workflows/tests.yml` (ubuntu-latest, py3.10) runs on every push; the full matrix
  arrives in Phase 5 (D-034). It tests **both** install paths: wheel (`pip install ".[test]"`) and
  editable with non-isolated build deps.
- Gotcha (hit on the first CI run): a meson-python editable install performed **with** build
  isolation breaks at import time — the editable loader rebuilds via
  `/tmp/pip-build-env-*/overlay/bin/ninja`, which no longer exists. Always editable-install with
  `--no-build-isolation` after installing `meson-python meson ninja cython numpy` in the target
  environment.
- Loop: `.clinerules/workflows/publish-progress.md` — push after every green task, merge to
  `main` only after a gate is approved.
- Git identity is repo-local: `S-B-Iqbal <28038429+S-B-Iqbal@users.noreply.github.com>` (D-031).

## Local machine (dev, 2026-12-11)
- macOS 14.3.1, Apple M2, 8 cores (4 performance + 4 efficiency), 64-bit arm64.
- Compiler: Apple clang 14.0.3 · BLAS: Apple Accelerate (numpy 2.2.6 wheel).
- `.venv`: CPython 3.10.9 · numpy 2.2.6 · scipy 1.15.3 · scikit-learn 1.7.2 ·
  threadpoolctl 3.8.0 · Cython 3.3.1 · meson 1.12.1 · meson-python 0.22.1 · pytest 9.2.0 ·
  ruff 0.17.4.
- Benchmarks on this machine will be single-socket laptop numbers; do not compare them to the
  paper's 64-core Xeon + Faiss CPU results (rule 09).
