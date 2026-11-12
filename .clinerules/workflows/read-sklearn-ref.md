# /read-sklearn-ref.md — study scikit-learn before Phase 1 code

1. If `_ref/scikit-learn` does not exist:
   `git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git _ref/scikit-learn`
   Record the commit hash: `git -C _ref/scikit-learn rev-parse HEAD`.
2. Read (do not modify):
   - `sklearn/cluster/_kmeans.py` (KMeans, MiniBatchKMeans, _BaseKMeans, _labels_inertia*,
     _mini_batch_step, _kmeans_plusplus, kmeans_plusplus, partial_fit, _check_params_vs_input)
   - `sklearn/cluster/_k_means_common.pyx/.pxd` (CHUNK_SIZE, _inertia_dense/_sparse, helpers)
   - `sklearn/cluster/_k_means_lloyd.pyx` (chunked prange + gemm + argmin, local buffers)
   - `sklearn/cluster/_k_means_minibatch.pyx` (_minibatch_update_dense/_sparse)
   - `sklearn/cluster/meson.build`, `sklearn/utils/_param_validation.py`, `sklearn/utils/validation.py`
   - `sklearn/cluster/tests/test_k_means.py` (test style for MiniBatchKMeans)
3. Write into `memory-bank/systemPatterns.md` → section "Observed in scikit-learn @<hash>":
   exact signatures, validation calls, attribute names, threadpool handling, reassignment rule,
   n_steps_ semantics, sample_weight handling, sparse handling.
4. Note anything conflicting with rules 03/04 as a new Q in openQuestions.md.
5. Never edit `_ref/`. Never copy large blocks verbatim — mirror structure and naming.
