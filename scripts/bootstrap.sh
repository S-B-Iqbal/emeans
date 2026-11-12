#!/usr/bin/env bash
# Local bootstrap for the emeans repo. Safe to re-run. Does NOT push anything.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}

# 1) git repo
[ -d .git ] || git init -b main

# 2) virtual environment + dev tooling
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install "numpy>=1.24" scipy "scikit-learn>=1.6" threadpoolctl \
  meson-python meson ninja "cython>=3" \
  pytest pytest-cov pre-commit ruff cython-lint \
  sphinx numpydoc sphinx-gallery pydata-sphinx-theme matplotlib pandas

# 3) read-only scikit-learn reference
if [ ! -d _ref/scikit-learn ]; then
  mkdir -p _ref
  git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git _ref/scikit-learn
fi
echo "sklearn ref commit: $(git -C _ref/scikit-learn rev-parse HEAD)"
chmod -R a-w _ref/scikit-learn || true   # make accidental edits fail loudly

# 4) paper
mkdir -p docs
if [ ! -f docs/e-means.pdf ]; then
  curl -L -o docs/e-means.pdf https://arxiv.org/pdf/2611.09501v1
fi

# 5) hooks
pre-commit install

echo "Done. Open the folder in VS Code, start Cline in PLAN mode, and say:"
echo "  'Read the memory bank and start Phase 0.'"
