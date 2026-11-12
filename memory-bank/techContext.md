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

## Local machine (dev, 2026-12-11)
- macOS 14.3.1, Apple M2, 8 cores (4 performance + 4 efficiency), 64-bit arm64.
- Compiler: Apple clang 14.0.3 · BLAS: Apple Accelerate (numpy 2.2.6 wheel).
- `.venv`: CPython 3.10.9 · numpy 2.2.6 · scipy 1.15.3 · scikit-learn 1.7.2 ·
  threadpoolctl 3.8.0 · Cython 3.3.1 · meson 1.12.1 · meson-python 0.22.1 · pytest 9.2.0 ·
  ruff 0.17.4.
- Benchmarks on this machine will be single-socket laptop numbers; do not compare them to the
  paper's 64-core Xeon + Faiss CPU results (rule 09).
