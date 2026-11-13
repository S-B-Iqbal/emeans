# SPDX-License-Identifier: BSD-3-Clause
"""The EMeans estimator (Algorithm 1 of arXiv:2611.09501v1)."""

from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp

from emeans._compat import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    Interval,
    StrOptions,
    TransformerMixin,
    _check_sample_weight,
    _fit_context,
    _openmp_effective_n_threads,
    check_array,
    check_is_fitted,
    check_random_state,
    kmeans_plusplus,
    validate_data,
)
from emeans._numpy_backend import (
    assign,
    distances,
    labels_inertia,
    reduce_batch,
)
from emeans._revival import revive
from emeans._scaling_laws import auto_alpha0, auto_batch_size
from emeans._schedule import (
    alpha as alpha_at,
)
from emeans._schedule import (
    n_updates,
)

# Sec. 3.1: every cluster starts at N_j = 1e-10 and S_j = N_j * c_j.
INIT_COUNT = 1e-10
# App. B.1: the laws were fitted with 1e5 <= n_samples <= 1e8, n/k >= 8.
MIN_SAMPLES_FOR_LAWS = 100_000
MIN_SAMPLES_PER_CLUSTER = 8


def _streaming_step(estimator):
    """Constant step of ``partial_fit`` when it continues a ``fit`` (D-042)."""
    if estimator.alpha0 != "auto":
        return float(estimator.alpha0)
    return estimator.alpha_floor_ratio * estimator.alpha0_


class EMeans(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    ClusterMixin,
    BaseEstimator,
):
    """e-means clustering with exponential-moving-average centroids.

    ``EMeans`` is an independent, scikit-learn compatible implementation of
    e-means (Bessa et al., "e-means: outperforming k-means (in one-tenth the
    epochs)", arXiv:2611.09501v1).  Every cluster keeps an EMA of the count
    and the sum of the rows assigned to it -- the centroid is ``S / N``
    (Eq. 4) -- updated with a step annealed on a cosine (Eq. 5), with dead
    clusters revived by splitting a donor (App. H.1), and finalised either on
    the last-epoch mean or on a slow EMA (Sec. 3.2).  The batch size (Eq. 7)
    and the peak step (Eq. 6) come from fitted scaling laws, so the only
    thing to choose is ``n_clusters``.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters ``k``.
    n_epochs : int, default=5
        Number of epochs ``T``, i.e. passes over the data (Sec. 3.1).
    batch_size : int or "auto", default="auto"
        Rows per EMA update.  ``"auto"`` resolves Eq. 7, rounded to the
        nearest power of two and capped at ``n_samples``.
    alpha0 : float or "auto", default="auto"
        Peak step of the cosine schedule, in (0, 1].  ``"auto"`` uses Eq. 6
        evaluated with the resolved batch size (D-006).
    alpha_floor_ratio : float, default=0.1
        ``epsilon`` of Eq. 5: the schedule ends at this fraction of alpha0.
    init : {"random", "k-means++"} or array-like, default="random"
        Initial centroids.  ``"random"`` draws distinct rows with
        probability proportional to ``sample_weight``; ``"k-means++"`` uses
        the greedy sklearn variant (App. G.2), much slower to compute.
    revival_threshold : float, default=0.1
        ``tau`` of Sec. 3.1: a cluster is dead when its count falls below
        ``tau`` times the mean count.
    revival_period : int, default=5
        Revive dead clusters every this many EMA updates (Alg. 1, line 9).
    finalization : {"last_epoch_mean", "slow_ema"} or None, default="last_epoch_mean"
        How the final centroids are built (Sec. 3.2): the mean of the rows
        assigned in the last epoch, a slow EMA that is never used for
        assignment, or ``None`` for the EMA centroids ``S / N``.
    slow_ema_speed : float, default=1.0
        Step of the slow EMA as a multiple of the cosine floor
        ``epsilon * alpha0`` (App. I, Table 15).
    random_state : int, RandomState instance or None, default=None
        Controls the row permutations, the initialisation and the revival.
    verbose : int, default=0
        If positive, print one line per epoch.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final centroids, in the dtype of the training data.
    labels_ : ndarray of shape (n_samples,), dtype int32
        Cluster of every training row, from one extra assignment pass with
        the final centroids, so that ``labels_ == predict(X)`` (D-018).
    inertia_ : float
        Sum of squared distances of the training rows to their centroid,
        weighted by ``sample_weight``.
    n_iter_ : int
        Number of epochs completed (``n_epochs`` after a full fit).
    n_steps_ : int
        EMA updates performed: ``T * ceil(n_samples / batch_size)`` after
        ``fit``, one per ``partial_fit`` call.
    batch_size_ : int
        Resolved batch size.
    alpha0_ : float
        Resolved peak step.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names, only when ``X`` has string column names.

    See Also
    --------
    sklearn.cluster.KMeans : Lloyd algorithm, many full passes.
    sklearn.cluster.MiniBatchKMeans : MacQueen-style mini-batch k-means.

    Notes
    -----
    The scaling laws were fitted on DEEP (d = 96) with 1e5 <= n_samples <=
    1e8, 256 <= n_clusters <= 65536 and n_samples / n_clusters >= 8
    (Sec. 4.1, App. B.1).  Outside that range the automatic batch size and
    peak step are an extrapolation -- notably on very low dimensional data
    (App. D.2) -- so a ``UserWarning`` is emitted (D-021) and explicit
    values are the safer choice.

    Distances are computed in float64 whatever the input dtype (D-037); the
    fitted state (counts and sums) is float64 as well, while
    ``cluster_centers_`` and ``transform`` follow the dtype of ``X``.

    ``partial_fit`` is an engineering extension that the paper does not
    evaluate: the step is held constant instead of annealed over an unknown
    horizon, and ``batch_size_`` records the first batch size (D-014).

    References
    ----------
    .. [1] S. Bessa, P.-E. Mazare, G. Szilvasy, H. Touvron, D. Picard,
       M. Douze and H. Jegou. "e-means: outperforming k-means (in
       one-tenth the epochs)". arXiv:2611.09501, 2026.

    Examples
    --------
    >>> import numpy as np
    >>> from emeans import EMeans
    >>> rng = np.random.RandomState(0)
    >>> centers = np.array([[-3.0, 0.0], [3.0, 0.0]])
    >>> X = np.vstack([c + 0.1 * rng.normal(size=(20, 2)) for c in centers])
    >>> model = EMeans(n_clusters=2, n_epochs=2, batch_size=16, alpha0=0.5,
    ...                random_state=0).fit(X)
    >>> model.labels_.shape
    (40,)
    >>> model.n_iter_
    2
    >>> model.predict(X).shape
    (40,)
    """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "n_epochs": [Interval(Integral, 1, None, closed="left")],
        "batch_size": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "alpha0": [
            StrOptions({"auto"}),
            Interval(Real, 0, 1, closed="right"),
        ],
        "alpha_floor_ratio": [Interval(Real, 0, 1, closed="neither")],
        "init": [StrOptions({"random", "k-means++"}), "array-like"],
        "revival_threshold": [Interval(Real, 0, 1, closed="right")],
        "revival_period": [Interval(Integral, 1, None, closed="left")],
        "finalization": [
            StrOptions({"last_epoch_mean", "slow_ema"}),
            None,
        ],
        "slow_ema_speed": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        n_epochs=5,
        batch_size="auto",
        alpha0="auto",
        alpha_floor_ratio=0.1,
        init="random",
        revival_threshold=0.1,
        revival_period=5,
        finalization="last_epoch_mean",
        slow_ema_speed=1.0,
        random_state=None,
        verbose=0,
    ):
        self.n_clusters = n_clusters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.alpha_floor_ratio = alpha_floor_ratio
        self.init = init
        self.revival_threshold = revival_threshold
        self.revival_period = revival_period
        self.finalization = finalization
        self.slow_ema_speed = slow_ema_speed
        self.random_state = random_state
        self.verbose = verbose

    def _check_params_vs_input(self, X):
        """Raise for n_samples < n_clusters; warn outside the fitted range."""
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}."
            )
        uses_auto = self.batch_size == "auto" or self.alpha0 == "auto"
        small_n = n_samples < MIN_SAMPLES_FOR_LAWS
        few_per_cluster = n_samples / self.n_clusters < MIN_SAMPLES_PER_CLUSTER
        if uses_auto and (small_n or few_per_cluster):
            warnings.warn(
                "EMeans scaling laws were fitted for 1e5 <= n_samples <= 1e8 "
                f"with n_samples / n_clusters >= {MIN_SAMPLES_PER_CLUSTER}; "
                f"here n_samples={n_samples} and n_clusters={self.n_clusters}, "
                "so the automatic batch size and peak step are extrapolated "
                "(pass batch_size and alpha0 explicitly to silence this).",
                UserWarning,
                stacklevel=2,
            )

    def _validate_center_shape(self, centers):
        """Check that an initial-centers array has shape (n_clusters, d)."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != self.n_features_in_:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {self.n_features_in_}."
            )

    def _resolve_hyperparameters(self, n_samples):
        """Resolve batch size (Eq. 7) then the peak step (Eq. 6, D-006)."""
        if self.batch_size == "auto":
            batch_size = auto_batch_size(n_samples, self.n_clusters, self.n_epochs)
        else:
            batch_size = min(int(self.batch_size), n_samples)
        if self.alpha0 == "auto":
            peak = auto_alpha0(n_samples, self.n_clusters, self.n_epochs, batch_size)
        else:
            peak = float(self.alpha0)
        return batch_size, peak

    def _init_centroids(self, X, random_state, sample_weight):
        """Initial centroids from random rows, k-means++ or a given array."""
        if isinstance(self.init, str) and self.init == "k-means++":
            centers, _ = kmeans_plusplus(
                X,
                self.n_clusters,
                sample_weight=sample_weight,
                random_state=random_state,
            )
        elif isinstance(self.init, str):  # "random": rows, p ~ w (D-026)
            probabilities = sample_weight / sample_weight.sum()
            indices = random_state.choice(
                X.shape[0],
                size=self.n_clusters,
                replace=False,
                p=probabilities,
            )
            centers = X[indices]
            if sp.issparse(centers):
                centers = centers.toarray()
        else:
            centers = check_array(self.init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(centers)
        return np.asarray(centers, dtype=np.float64)

    def _fit_emeans(
        self, X, sample_weight, counts, sums, batch_size, peak, random_state
    ):
        """Run the EMA loop (Alg. 1 lines 3-11) and store the final state."""
        n_samples = X.shape[0]
        total_steps = n_updates(n_samples, batch_size, self.n_epochs)
        use_slow = self.finalization == "slow_ema"
        slow_counts = slow_sums = None
        slow_step = 0.0
        if use_slow:
            slow_counts = counts.copy()
            slow_sums = sums.copy()
            slow_step = self.slow_ema_speed * self.alpha_floor_ratio * peak
        record_last = self.finalization == "last_epoch_mean"
        last_labels = None
        if record_last:
            last_labels = np.empty(n_samples, dtype=np.int32)
        step = peak
        t = 0
        for epoch in range(self.n_epochs):
            order = random_state.permutation(n_samples)
            epoch_inertia = 0.0
            revived = 0
            for start in range(0, n_samples, batch_size):
                stop = min(start + batch_size, n_samples)
                indices = order[start:stop]
                t += 1
                centers = sums / counts[:, None]
                if self.verbose:
                    labels, batch_inertia = labels_inertia(
                        X[indices], centers, sample_weight[indices]
                    )
                    epoch_inertia += batch_inertia
                else:
                    labels = assign(X[indices], centers)
                batch_counts, batch_sums = reduce_batch(
                    X[indices], labels, self.n_clusters, sample_weight[indices]
                )
                step = alpha_at(t, total_steps, peak, self.alpha_floor_ratio)
                counts *= 1.0 - step
                counts += step * batch_counts
                sums *= 1.0 - step
                sums += step * batch_sums
                if use_slow:
                    slow_counts *= 1.0 - slow_step
                    slow_counts += slow_step * batch_counts
                    slow_sums *= 1.0 - slow_step
                    slow_sums += slow_step * batch_sums
                if record_last and epoch == self.n_epochs - 1:
                    last_labels[indices] = labels
                if t % self.revival_period == 0:
                    states = [(counts, sums)]
                    if use_slow:
                        states.append((slow_counts, slow_sums))
                    revived += revive(states, random_state, self.revival_threshold)
            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs}: steps {t}/{total_steps}, "
                    f"alpha {step:.4f}, epoch inertia {epoch_inertia:.6g}, "
                    f"revived {revived}"
                )
        self._counts = counts
        self._sums = sums
        self._slow_counts = slow_counts
        self._slow_sums = slow_sums
        self._last_epoch_labels = last_labels
        self.n_steps_ = t

    def _final_centroids(self, X, sample_weight):
        """Final centroids for the selected finalization (Sec. 3.2)."""
        fast = self._sums / self._counts[:, None]
        if self.finalization == "slow_ema":
            return self._slow_sums / self._slow_counts[:, None]
        if self.finalization == "last_epoch_mean":
            return self._last_epoch_mean(X, sample_weight, fast)
        return fast

    def _last_epoch_mean(self, X, sample_weight, fallback):
        """Average the rows of the last epoch per cluster (D-009)."""
        counts, sums = reduce_batch(
            X, self._last_epoch_labels, self.n_clusters, sample_weight
        )
        means = np.array(fallback, dtype=np.float64, copy=True)
        filled = counts > 0.0
        means[filled] = sums[filled] / counts[filled, None]
        return means

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """Compute the e-means centroids on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training rows; CSR input is accepted.
        y : Ignored
            Present for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-row weights; ``None`` gives every row unit weight.

        Returns
        -------
        self : EMeans
            Fitted estimator.
        """
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        sample_weight = np.asarray(
            _check_sample_weight(sample_weight, X, dtype=X.dtype), dtype=np.float64
        )
        self._n_threads = _openmp_effective_n_threads()
        n_samples = X.shape[0]
        self.batch_size_, self.alpha0_ = self._resolve_hyperparameters(n_samples)
        self._partial_fit_step = _streaming_step(self)
        centers = self._init_centroids(X, random_state, sample_weight)
        counts = np.full(self.n_clusters, INIT_COUNT, dtype=np.float64)
        sums = counts[:, None] * centers
        self._fit_emeans(
            X,
            sample_weight,
            counts,
            sums,
            self.batch_size_,
            self.alpha0_,
            random_state,
        )
        final = self._final_centroids(X, sample_weight)
        self.cluster_centers_ = final.astype(X.dtype, copy=False)
        self._n_features_out = self.cluster_centers_.shape[0]
        self.n_iter_ = self.n_epochs
        # One extra pass so that labels_ matches the returned centroids (D-018).
        self.labels_, self.inertia_ = labels_inertia(
            X, self.cluster_centers_, sample_weight
        )
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def _check_test_data(self, X):
        """Validate new data against the fitted feature count."""
        return validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            reset=False,
        )

    def _transform(self, X):
        """Cluster distances without re-validating X (sklearn pattern)."""
        squared = distances(X, self.cluster_centers_)
        return np.sqrt(squared).astype(X.dtype, copy=False)

    def predict(self, X):
        """Predict the closest cluster for each row of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New rows to assign.

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int32
            Index of the closest centroid for each row.
        """
        check_is_fitted(self)
        return assign(self._check_test_data(X), self.cluster_centers_)

    def transform(self, X):
        """Distances from each row of X to every cluster centre.

        In the new space each column is the Euclidean distance (not squared)
        to one centroid, so the output dtype follows the input dtype.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
        """
        check_is_fitted(self)
        return self._transform(self._check_test_data(X))

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit the model and return the labels of the training rows."""
        return self.fit(X, sample_weight=sample_weight).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit the model and transform X to the cluster-distance space."""
        return self.fit(X, sample_weight=sample_weight).transform(X)

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the inertia of X with respect to the fitted centroids.

        Returns
        -------
        score : float
            ``-sum_i w_i * ||x_i - c_{a(i)}||**2``, so higher is better.
        """
        check_is_fitted(self)
        X = self._check_test_data(X)
        sample_weight = np.asarray(
            _check_sample_weight(sample_weight, X, dtype=X.dtype),
            dtype=np.float64,
        )
        _, inertia = labels_inertia(X, self.cluster_centers_, sample_weight)
        return -inertia

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, sample_weight=None):
        """Update the model with a single mini-batch (streaming extension).

        The step is held **constant** across calls (there is no horizon to
        anneal over): a float ``alpha0`` is used verbatim, while ``"auto"``
        freezes ``alpha_floor_ratio * alpha0_law(k, B, n, T=1)`` on the first
        call (D-014).  The first call must supply at least ``n_clusters``
        rows, since its rows seed the initial centroids (D-036).

        Returns
        -------
        self : EMeans
        """
        has_centers = hasattr(self, "cluster_centers_")
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            reset=not has_centers,
        )
        self._random_state = getattr(
            self, "_random_state", check_random_state(self.random_state)
        )
        sample_weight = np.asarray(
            _check_sample_weight(sample_weight, X, dtype=X.dtype), dtype=np.float64
        )
        if not has_centers:
            if X.shape[0] < self.n_clusters:
                raise ValueError(
                    f"n_samples={X.shape[0]} should be >= n_clusters="
                    f"{self.n_clusters} on the first partial_fit call."
                )
            self._n_threads = _openmp_effective_n_threads()
            n_rows = X.shape[0]
            self.batch_size_ = n_rows
            if self.alpha0 == "auto":
                peak = auto_alpha0(n_rows, self.n_clusters, 1, n_rows)
                self.alpha0_ = peak
                self._partial_fit_step = self.alpha_floor_ratio * peak
            else:
                self.alpha0_ = float(self.alpha0)
                self._partial_fit_step = float(self.alpha0)
            centers = self._init_centroids(X, self._random_state, sample_weight)
            self._counts = np.full(self.n_clusters, INIT_COUNT, dtype=np.float64)
            self._sums = self._counts[:, None] * centers
            self.n_steps_ = 0
            self.n_iter_ = 0
        else:
            self.n_steps_ = getattr(self, "n_steps_", 0)
            if not hasattr(self, "_partial_fit_step"):
                self._partial_fit_step = _streaming_step(self)
        step = self._partial_fit_step
        centers = self._sums / self._counts[:, None]
        labels = assign(X, centers)
        batch_counts, batch_sums = reduce_batch(
            X, labels, self.n_clusters, sample_weight
        )
        self._counts *= 1.0 - step
        self._counts += step * batch_counts
        self._sums *= 1.0 - step
        self._sums += step * batch_sums
        self.n_steps_ += 1
        if self.n_steps_ % self.revival_period == 0:
            revive(
                [(self._counts, self._sums)],
                self._random_state,
                self.revival_threshold,
            )
        self.cluster_centers_ = (self._sums / self._counts[:, None]).astype(
            X.dtype, copy=False
        )
        self._n_features_out = self.cluster_centers_.shape[0]
        self.labels_, self.inertia_ = labels_inertia(
            X, self.cluster_centers_, sample_weight
        )
        return self
