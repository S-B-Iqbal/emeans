# References — what each source contributes

1. **Paper** — `docs/e-means.pdf`, https://arxiv.org/abs/2611.09501 (HTML: /html/2611.09501v1).
   NORMATIVE. Eq. 1–8, Alg. 1, Tables 1–2, Sec. 3.2 finalization, App. B laws, App. H revival,
   App. I slow EMA, Sec. 4.1 protocol. Digest: `docs/PAPER_NOTES.md`.
2. **Faiss contrib** — https://github.com/facebookresearch/faiss/tree/main/contrib/e_means
   (shipped in Faiss 1.16.0; PRs #5812, #5842, #5854). BEHAVIOURAL ORACLE ONLY — DO NOT COPY CODE.
   Notable divergences from the paper (record, don't adopt without approval):
   batch laws "throughput"/"quality"/"auto" with 6-sig-fig constants; B capped ≈ n/2;
   quality law at d ≤ 16 and revival off at d ≤ 2 (PR #5854); last partial batch kept;
   labels returned from the last training epoch; no partial_fit; dense float32 only.
3. **Website** — https://e-means.github.io/ — explanations and figures: Lloyd/MacQueen/EMA update
   rules (Lloyd as the B=n, α=1 special case), EMA vs SGD convex-hull argument, revival comparison,
   BibTeX for the README.
4. **X post** — https://x.com/swann_bessa/status/2120201545426088021 — positioning: drop-in k-means
   replacement, no tuning beyond k, released in Faiss 1.16.0. Secondary; no technical spec beyond the paper.
5. **scikit-learn** — `_ref/scikit-learn` (read-only): `sklearn/cluster/_kmeans.py`,
   `_k_means_common.pyx`, `_k_means_lloyd.pyx`, `_k_means_minibatch.pyx`; developer guide
   (https://scikit-learn.org/stable/developers/develop.html); FAQ inclusion criteria
   (https://scikit-learn.org/stable/faq.html).
6. **Upstream** — scikit-learn issue #35383 "Add e-means clustering (EMeans)";
   scikit-learn-contrib workflow: https://github.com/scikit-learn-contrib/scikit-learn-contrib
