# SPDX-License-Identifier: BSD-3-Clause
"""scikit-learn estimator checks (Phase 2, task P2.1c)."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from emeans import EMeans

# Justified failures (D-041): e-means samples mini-batches at random, reshuffles
# every epoch, anneals the step on a cosine and revives dead clusters, while its
# batch size and peak step depend on n_samples.  Replicating a row therefore
# changes the trajectory, so integer weights are not equivalent to repeated rows.
EXPECTED_FAILED_CHECKS = {
    "check_sample_weight_equivalence_on_dense_data": (
        "integer weights are not equivalent to repeated rows: e-means samples"
        " mini-batches at random and its hyper-parameters depend on n_samples"
        " (D-041)"
    ),
    "check_sample_weight_equivalence_on_sparse_data": (
        "same reason as the dense case: mini-batch sampling plus n-dependent"
        " hyper-parameters (D-041)"
    ),
}


@parametrize_with_checks(
    [
        EMeans(n_clusters=3, n_epochs=2, batch_size=16, alpha0=0.5, random_state=0),
        EMeans(n_clusters=3, random_state=0),
    ],
    expected_failed_checks=lambda estimator: EXPECTED_FAILED_CHECKS,
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
