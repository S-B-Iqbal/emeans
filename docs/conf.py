# SPDX-License-Identifier: BSD-3-Clause
"""Sphinx configuration for the emeans documentation (Phase 5, P5.2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "emeans"
copyright = "2026, S-B-Iqbal and contributors"
author = "S-B-Iqbal"
release = "0.1.0.dev0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
numpydoc_show_class_members = False
autodoc_typehints = "description"

html_theme = "pydata_sphinx_theme"
exclude_patterns = [
    "_build",
    "auto_examples",
    "e-means.pdf",
    "PAPER_NOTES.md",
    "REFERENCES.md",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "auto_examples",
    "gallery_dirs": "generated/examples",
    "filename_pattern": r"/plot_",
    "plot_gallery": True,
}
