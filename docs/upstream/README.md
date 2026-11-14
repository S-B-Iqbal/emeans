# Upstream drafts (Phase 6)

Drafts prepared for scikit-learn and scikit-learn-contrib. **Nothing in this directory has been
posted anywhere** (rules 01 and 10): no issue comment, no PR, no Discussion, no email.

These are Markdown files on purpose, so Sphinx never picks them up: they must not appear in the
published documentation until the user approves posting, and `docs/index.rst` deliberately does
not link to them.

| File | Task | Target when approved |
|---|---|---|
| `parity-matrix.md` | P6.0 | reference for every other draft; keep private |
| `contrib-request.md` | P6.1 | new issue in `scikit-learn-contrib/scikit-learn-contrib` |
| `sklearn-35383-comment.md` | P6.2 | comment on scikit-learn issue #35383 |
| `discussion-minibatch-improvements.md` | P6.3 | scikit-learn Discussion (or a follow-up comment) |
| `minibatch-pr-sketch.md` | P6.4 | design input for a future `MiniBatchKMeans` PR |
| `example-embeddings-sketch.md` | P6.5 | `examples/cluster/` in scikit-learn |

Ground rules used to write them (rule 09 and rule 10):

- every number is traceable to `benchmarks/results/suite.csv`;
- our wall-clock numbers come from a **serial** build and are never compared with OpenMP wheels;
- the losing cases are in every draft that quotes numbers;
- no superlatives, no claim of novelty, and no text copied from the paper, from scikit-learn or
  from Faiss.
