# Tech context

- Python 3.10–3.13 · numpy · scipy · scikit-learn>=1.6 · threadpoolctl
- Build: meson-python + meson + ninja + Cython>=3 (Phase 3 adds extensions); flat layout.
- Dev install: `pip install -e . --no-build-isolation` (meson-python, ninja, cython, numpy pre-installed)
- Tests: `pytest --pyargs emeans`, `pytest --doctest-modules emeans`
- Lint: pre-commit → ruff-check, ruff-format, cython-lint, basic hygiene hooks
- Docs: sphinx, numpydoc, sphinx-gallery, pydata-sphinx-theme
- CI (Phase 5): GitHub Actions matrix ubuntu/macos/windows × 3.10–3.13; wheels via cibuildwheel;
  macOS wheels vendor libomp; nightly-sklearn job.
- Read-only reference: `_ref/scikit-learn` (shallow clone; record commit hash here).
- Local machine notes: (fill in — OS, CPU, cores, BLAS)
