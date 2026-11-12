# 10 — Upstream path and licensing

- License: BSD-3-Clause. We copy no Faiss code (Faiss is MIT); we cite the paper and link Faiss.
- README: citation + BibTeX for arXiv:2611.09501 and a clear "independent implementation, not
  affiliated with Meta" note.
- scikit-learn: do NOT open a new-estimator PR or a duplicate issue. Existing issue
  https://github.com/scikit-learn/scikit-learn/issues/35383 ("Add e-means clustering (EMeans)").
  Phase 6 drafts (saved to `docs/upstream/`, NEVER posted by Cline):
  1. comment for #35383 (independent CPU benchmarks + contrib package link),
  2. Discussion text: would maintainers accept MiniBatchKMeans improvements — cosine-annealed EMA
     updates, split-based reassignment, last-epoch-mean finalization, auto batch size & step —
     framed under the FAQ "clear-cut improvement to a widely-used method" clause,
  3. sketch of the MiniBatchKMeans PR (files touched, new/changed params, back-compat plan).
- scikit-learn-contrib: draft the inclusion request (check_estimator passes, tests, PEP8/ruff, CI, docs).
