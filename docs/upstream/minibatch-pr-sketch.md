# `MiniBatchKMeans` PR sketch (P6.4, draft — never opened)

A sketch a reviewer could implement from. Target: `sklearn/cluster`. Draft only (rule 01/10).
Companion: `docs/upstream/discussion-minibatch-improvements.md` (P6.3, the rationale) and
`docs/upstream/parity-matrix.md` (P6.0, the evidence).

## 1. Public API (proposed)

```python
MiniBatchKMeans(
    n_clusters=8,
    *,
    init="k-means++",
    max_iter=100,
    batch_size=1024,
    verbose=0,
    compute_labels=True,
    random_state=None,
    tol=0.0,
    max_no_improvement=10,
    init_size=None,
    n_init="auto",
    reassignment_ratio=0.01,
    # new, all defaulting to today's behaviour
    ema_schedule="none",          # {"none", "cosine"}
    alpha0=None,                  # float in (0, 1] or "auto" (needs batch_size too)
    alpha_floor_ratio=0.1,        # the paper's epsilon
    finalization="last_batch",    # {"last_batch", "last_epoch_mean"}
    revival="none",               # {"none", "split"}
    revival_threshold=0.1,
    revival_period=5,
)
```

- `ema_schedule="none"` keeps the current count-based running mean, so **no behaviour change** and
  no test churn for existing users.
- `alpha0="auto"` / `batch_size="auto"` are only allowed together with `ema_schedule="cosine"`
  (the laws of Eq. 6–7 are defined for that setting) and warn when `n`, `n/k` or `d` are outside
  the fitted range (`1e5 ≤ n ≤ 1e8`, `n/k ≥ 8`, `d = 96` in the paper, App. B.1).
- With `ema_schedule="cosine"`, the state must become counts-and-sums (`N`, `S`) so that
  `c = S/N` is well defined after a step that no longer normalises by the batch; `_counts`
  already exists and can carry `N`, with a new `_sums` (k, d) array mirroring it. The existing
  `weight_sums` kernel argument becomes `N`.

## 2. Code changes

| File | Change |
|---|---|
| `sklearn/cluster/_kmeans.py` | new params in `__init__` (as attributes, verbatim), `_parameter_constraints` entries, validation in `_check_params_vs_input`/`_validate_params`, `_mini_batch_step` learns `alpha` (a scalar step) and, when `finalization="last_epoch_mean"`, records per-cluster `(count, sum)` of the final epoch; `_mini_batch_convergence` unchanged; `revival="split"` calls a new `_split_dead_clusters(...)` next to the existing reassignment block |
| `sklearn/cluster/_kmeans.py` (finalization) | after the loop: `centers = means` from the recorded final-epoch accumulators instead of `centers`, then the usual `compute_labels` pass — which then returns the same labels by construction (`fit(X).labels_ == predict(X)`) |
| `sklearn/cluster/_k_means_minibatch.pyx` / `.pxd` | optional `alpha` argument: when set, `centers_new[j] = (1−alpha)·centers_old[j] + alpha·batch_mean_j` (empty clusters keep the old centre); keep the current path untouched when `alpha` is NULL |
| `sklearn/cluster/tests/test_kmeans.py` | new tests, see §4 |
| `sklearn/utils/extmath.py` | none (no new math helpers) |
| `doc/modules/clustering.rst` | short paragraph: what the schedule and the finalization change, with the honest "no measured benefit below ~1e5 rows, higher dimensions benefit most" caveat |
| `doc/whats_new/v1.11.rst` | entry under `sklearn.cluster` |
| `benchmarks/` | extend the existing MiniBatchKMeans benchmark script rather than adding a new one |

Kernel notes: the new step keeps the OpenMP structure (`prange` over clusters, per-thread
buffers) and stays dtype-generic through the existing `floating` fused type; `alpha` is one scalar
per step, so no atomics are needed.

## 3. Semantics to pin down in the PR description

1. `M` (the schedule horizon) = `max_iter · ceil(n_samples / batch_size)`; with
   `max_no_improvement` early stopping the schedule stops where it stopped, and `finalization`
   still uses the recorded final epoch.
2. `sample_weight` enters both the EMA update and the final-epoch mean (weighted), consistent with
   the existing weighted assignment.
3. `alpha0="auto"` is computed from the *resolved* batch size (Eq. 6 depends on Eq. 7).
4. Revival is `off` by default: `reassignment_ratio` keeps working exactly as today, and the two
   mechanisms are documented as mutually exclusive for now.

## 4. Test plan

- `ema_schedule="none"` with the same seed reproduces today's `cluster_centers_` **bit for bit**
  (regression guard for the default path).
- `alpha=0.5` for a single batch equals an explicit EMA computed in the test.
- `finalization="last_epoch_mean"` ⇒ train inertia ≤ the `last_batch` inertia on the same run, and
  `labels_ == predict(X)` exactly.
- `alpha0="auto"` ⇒ `batch_size` is a power of two, `batch_size ≤ n_samples`, `alpha0 ≤ 1`, and a
  warning fires for `n = 500` (outside the fitted range).
- `revival="split"`: Σ`N` unchanged, both children have `N_donor/2`, `S == c·N` after the split,
  donor drawn proportionally to `N` (statistical test over many draws).
- `parametrize_with_checks([MiniBatchKMeans(...)])` on each new configuration.

## 5. Back-compatibility and release notes

- Every new parameter's default preserves current numerics; no deprecation needed.
- Old signatures keep working (contrib/`workflow.md` rule) — we add keyword-only params only.
- Reviewer question to answer in the PR: whether `finalization="last_epoch_mean"` should instead
  be the default for a major release, given it is provably not worse on train MSE.

## 6. Open questions before opening any PR

- [ ] Does the maintainers' preferred path (from #35383) put this work in `MiniBatchKMeans` at all,
      or in a separate estimator with `algorithm="e-means"` (Q-032)?
- [ ] Is a `benchmarks/` extension or a PR-dashboard run expected for a change of this size?
- [ ] User's explicit go-ahead (rule 01).
