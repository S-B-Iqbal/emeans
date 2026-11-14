# Request for inclusion in scikit-learn-contrib (P6.1, draft — never filed)

Target: a new issue in <https://github.com/scikit-learn-contrib/scikit-learn-contrib/issues/new>.
Status: **draft only** (rule 01: no write action on another repository without explicit
confirmation). Requirements below are quoted from `scikit-learn-contrib/workflow.md`, fetched
2026-11-14.

## The project

- **Repository:** <https://github.com/S-B-Iqbal/emeans> (public, BSD-3-Clause).
- **PyPI:** `emeans-cluster` (import name `emeans`) — *not yet released*; the name `emeans` on
  PyPI belongs to an unrelated Rust implementation, hence the distribution suffix (D-052).
- **What it is:** an independent, scikit-learn compatible implementation of e-means
  (Bessa et al., [arXiv:2611.09501](https://arxiv.org/abs/2611.09501)) written from the paper, with
  dense and CSR input, `sample_weight`, streaming `partial_fit`, three finalization strategies and
  compiled Cython/OpenMP kernels plus a NumPy reference backend.
- **Maintainer:** `S-B-Iqbal` (single maintainer today).

## Requirement checklist (`workflow.md`)

| Requirement | Status | Evidence |
|---|---|---|
| scikit-learn compatible (`check_estimator` passed) | **met** | `emeans/tests/test_common.py` runs `parametrize_with_checks` for explicit and `"auto"` configurations: green with 4 documented xfails (D-041) — `parametrize_with_checks` green in CI |
| Available on GitHub | **met** | public repository, MIT-style history in conventional commits |
| Open-source licence (BSD preferred) | **met** | BSD-3-Clause (`LICENSE`) |
| Documentation: guide, API reference, example gallery | **partial** | API reference (`docs/api.rst`) and an executed sphinx-gallery example (`docs/auto_examples/plot_emeans_quickstart.py`) exist and build with `-W -n`; **a narrative user guide is missing**, and the HTML is built in CI but not published to `gh-pages` |
| Unit tests | **met** | 269 tests + 271 doctests, 4 xfailed, coverage gate ≥ 90 % in CI |
| Python 3 compatible | **met** | Python ≥ 3.10, tested on 3.10–3.13 |
| PEP8 compliant | **met** | ruff-check + ruff-format (line length 88), clean in `pre-commit` and CI |
| Continuous integration | **met** | GitHub Actions: 3 OS × Python 3.10–3.13, docs job, coverage job, weekly nightly-scikit-learn job, cibuildwheel wheels built as artefacts |

Known gaps, stated up front: no PyPI release yet, no `gh-pages` documentation, no user guide, and
Windows/macOS wheels are serial because their toolchains have no OpenMP (D-044, D-056).

## The duplicate-algorithm question (needs the org's answer)

`sklearn-contrib-emeans` (0.3.1, `cakedev0`) is already on PyPI and is being proposed for the same
organisation, so this request is **not** "add the first e-means project":

1. If the org wants a single e-means project, we will not compete for the slot: the useful
   outcome is then to fold our two missing capabilities into that project — `sample_weight` for
   `fit`/`partial_fit` and streaming `partial_fit` (Q-032) — and to keep ours as an independent
   reference implementation (see `docs/upstream/parity-matrix.md`).
2. If the org accepts more than one implementation, we would need a distinct distribution name,
   because `workflow.md` requires the PyPI name `sklearn-contrib-<project>` and
   `sklearn-contrib-emeans` is taken. A name such as `sklearn-contrib-emeans-reference` would work.
3. On acceptance we would also: transfer the repository to the organisation and fork it back as
   `workflow.md` describes, publish the Sphinx HTML to `gh-pages`, and keep old public signatures
   supported for two releases.

## Ready-to-paste issue body

```text
Request for inclusion: emeans-cluster (e-means clustering, independent implementation)

Repository: https://github.com/S-B-Iqbal/emeans
Licence: BSD-3-Clause · Python >= 3.10 · scikit-learn >= 1.6
Distribution: emeans-cluster on PyPI (import name: emeans)

EMeans implements e-means (Bessa et al., arXiv:2611.09501) written from the paper: EMA
counts/sums state, cosine-annealed step, batch size and peak step from the paper's scaling
laws, dead-cluster revival by donor splitting, and last-epoch-mean / slow-EMA finalization.

Checklist: check_estimator green (4 documented xfails), unit tests 269 + 271 doctests,
coverage >= 90% enforced, ruff-clean, Sphinx docs building with -W -n, CI on 3 OS x Python
3.10-3.13, BSD-3 licence, public repo.

Where this implementation differs from sklearn-contrib-emeans (already proposed for the
organisation): it supports sample_weight in fit/partial_fit, streaming partial_fit, three
finalization modes, and exposes the revival/schedule knobs for ablations.

Two things I would like the maintainers to decide:
1. whether the organisation wants a second e-means project at all, and
2. if yes, which distribution name to use (`sklearn-contrib-emeans` is taken).

Happy to fold sample_weight / partial_fit into the existing project instead if that is the
preferred outcome.
```

## Verification before filing (not yet done)

- [ ] `pip install emeans-cluster` works from a published release (no release exists yet).
- [ ] Sphinx HTML published to `gh-pages` for review.
- [ ] User guide page written (`docs/guide.rst`) and added to the toctree.
- [ ] User's explicit go-ahead to file the issue (rule 01).
