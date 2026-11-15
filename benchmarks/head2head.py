# SPDX-License-Identifier: BSD-3-Clause
"""Head-to-head against ``sklearn-contrib-emeans`` (P9.1).

Fits this package's ``EMeans`` and the independently released
``sklearn-contrib-emeans`` on the same in-range data sets, with the same epoch budget,
the same initialisation and a single thread for both, and records the test MSE relative
to Lloyd@25 alongside the wall-clock.

Usage (single thread for both, so the comparison is fair)::

    OMP_NUM_THREADS=1 python benchmarks/head2head.py

Writes ``benchmarks/results/head2head.csv``.  The other implementation is optional: if
``sklearn_emeans`` is not installed the script says so and stops.  Only its public API
is called; no code is copied from it.
"""

import csv
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import held_out_split, make_anisotropic, make_blobs  # noqa: E402
from harness import lloyd_at_25, mse, relative_mse  # noqa: E402

from emeans import EMeans  # noqa: E402

try:  # optional: the released package under comparison
    import sklearn_emeans
    from sklearn_emeans import EMeans as TheirEMeans
except ImportError:  # pragma: no cover - depends on the environment
    sklearn_emeans = None
    TheirEMeans = None

RESULTS = Path(__file__).resolve().parent / "results"

CONFIGS = [
    ("blobs-96d-n100k-k512", lambda: make_blobs(120_000, 96, 512, seed=1), 512),
    ("blobs-32d-n100k-k512", lambda: make_blobs(120_000, 32, 512, seed=1), 512),
    ("aniso-96d-n100k-k512", lambda: make_anisotropic(120_000, 96, 512, seed=2), 512),
]

SEEDS = list(range(5))
EPOCHS = 5

FIELDS = [
    "dataset",
    "n_train",
    "n_clusters",
    "n_features",
    "seed",
    "method",
    "epochs",
    "test_mse",
    "rel_test_mse",
    "seconds",
    "batch_size",
]


def _row(
    name, X_train, n_clusters, seed, method, epochs, test, ref, seconds, batch_size
):
    return {
        "dataset": name,
        "n_train": X_train.shape[0],
        "n_clusters": n_clusters,
        "n_features": X_train.shape[1],
        "seed": seed,
        "method": method,
        "epochs": epochs,
        "test_mse": test,
        "rel_test_mse": relative_mse(test, ref),
        "seconds": seconds,
        "batch_size": batch_size,
    }


def run(out_path):
    rows = []
    for name, factory, n_clusters in CONFIGS:
        X = factory()
        X_train, _, X_test = held_out_split(X, seed=0)
        for seed in SEEDS:
            centres, _ = lloyd_at_25(X_train, n_clusters, seed)
            ref = mse(X_test, centres)
            rows.append(
                _row(
                    name, X_train, n_clusters, seed, "lloyd@25", 25, ref, ref, 0.0, None
                )
            )

            models = [("emeans (ours)", EMeans)]
            if TheirEMeans is not None:
                models.append(("sklearn_contrib_emeans", TheirEMeans))
            for label, cls in models:
                model = cls(n_clusters=n_clusters, n_epochs=EPOCHS, random_state=seed)
                start = time.perf_counter()
                model.fit(X_train)
                seconds = time.perf_counter() - start
                test = mse(X_test, model.cluster_centers_)
                rows.append(
                    _row(
                        name,
                        X_train,
                        n_clusters,
                        seed,
                        label,
                        EPOCHS,
                        test,
                        ref,
                        seconds,
                        getattr(model, "batch_size_", None),
                    )
                )
        print(f"done {name}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarise(rows):
    """Print median test MSE (relative to Lloyd@25) and wall-clock per method."""
    methods = sorted({row["method"] for row in rows} - {"lloyd@25"})
    for name in sorted({row["dataset"] for row in rows}):
        print(f"== {name}")
        for method in methods:
            runs = [
                row
                for row in rows
                if row["dataset"] == name and row["method"] == method
            ]
            if not runs:
                continue
            rel = statistics.median(float(r["rel_test_mse"]) for r in runs)
            sec = statistics.median(float(r["seconds"]) for r in runs)
            batch = statistics.median(
                int(r["batch_size"]) for r in runs if r["batch_size"] is not None
            )
            print(
                f"   {method:24s} rel_med={rel:+.4f} s_med={sec:6.2f} batch_med={batch}"
            )


def main():
    warnings.simplefilter("ignore", UserWarning)  # the D-021 range notice
    if TheirEMeans is None:
        raise SystemExit(
            "sklearn-contrib-emeans is not installed; "
            "pip install 'sklearn-contrib-emeans==0.3.2' to run the comparison."
        )
    version = getattr(sklearn_emeans, "__version__", "unknown")
    print(f"comparing against sklearn-contrib-emeans {version}")
    print(f"OMP_NUM_THREADS={__import__('os').environ.get('OMP_NUM_THREADS', 'unset')}")
    rows = run(RESULTS / "head2head.csv")
    summarise(rows)
    print(f"wrote {RESULTS / 'head2head.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
