# SPDX-License-Identifier: BSD-3-Clause
"""Run the benchmark suite and write ``benchmarks/results/*.csv`` (P4.1-P4.2, P7.1).

Usage: python benchmarks/run_suite.py [quick|suite|large]

- ``quick`` writes ``results/quick.csv`` (smoke test, 1 seed).
- ``suite`` writes ``results/suite.csv``: the original four datasets at
  ``n_train = 10 000``, i.e. **below** the range the scaling laws were fitted on.
- ``large`` writes ``results/large.csv``: ``n_train = 100 000``, ``d = 32/96`` and
  ``k = 512``, inside the fitted range (``1e5 <= n <= 1e8``, Sec. 4.1 / App. B.1),
  with a wider MiniBatchKMeans sweep.

Every configuration records the Lloyd@25 reference (rule 09), EMeans at integer
epoch budgets and the MiniBatchKMeans sweep selected on validation.  EMeans uses the
automatic hyper-parameters and is never tuned.
"""

import csv
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import held_out_split, make_blobs, make_high_n_over_k, make_low_dimension
from harness import emeans_at, lloyd_at_25, minibatch_sweep, mse, relative_mse

RESULTS = Path(__file__).resolve().parent / "results"

CONFIGS = [
    ("blobs-32d-k512", lambda: make_blobs(20000, 32, 512, seed=0), 512, 5),
    ("blobs-96d-k512", lambda: make_blobs(20000, 96, 512, seed=1), 512, 5),
    ("high-n-over-k", lambda: make_high_n_over_k(20000, 20, seed=2), 20, 5),
    ("low-d", lambda: make_low_dimension(20000, seed=3), 32, 5),
]

QUICK = [
    ("blobs-32d-k64", lambda: make_blobs(4000, 32, 64, seed=0), 64, 2),
]

# 120 000 points minus the 20 000 held out gives n_train = 100 000 and k = 512, so
# n / k = 195 >= 8 and n is inside the range the Table 2 laws were fitted on.  The
# resolver returns B = 8 192 and alpha0 = 0.733 here, against B = 1 024 and
# alpha0 = 0.318 at n_train = 10 000.
LARGE = [
    ("blobs-96d-n100k-k512", lambda: make_blobs(120_000, 96, 512, seed=1), 512, 5),
    ("blobs-32d-n100k-k512", lambda: make_blobs(120_000, 32, 512, seed=1), 512, 5),
]

# MiniBatchKMeans sweeps: the batch sizes must scale with n for the comparison to be
# fair at 1e5 (the paper sweeps up to 524 288 at 1e9 scale, App. C).
SWEEP_BATCH_SIZES = {
    "quick": (256, 1024),
    "suite": (256, 1024, 4096),
    "large": (1024, 8192, 65536),
}

MODES = {
    "quick": (QUICK, [0], "quick", False),
    "suite": (CONFIGS, list(range(5)), "suite", False),
    "large": (LARGE, list(range(5)), "large", True),
}


FIELDS = [
    "dataset",
    "n_train",
    "n_clusters",
    "n_features",
    "seed",
    "method",
    "epochs",
    "val_mse",
    "test_mse",
    "rel_test_mse",
    "seconds",
]


def run(
    configs,
    seeds,
    out_path,
    batch_sizes=(256, 1024, 4096),
    minibatch_per_budget=False,
):
    """Run every configuration and seed, and return the list of CSV rows.

    ``minibatch_per_budget`` sweeps MiniBatchKMeans once per epoch budget instead of
    once at ``max_epochs``, so its rows can be compared with EMeans at equal epochs
    (P7.1).  It multiplies the sweep cost by ``max_epochs``, so the original
    ``suite`` numbers are left untouched.
    """
    rows = []
    for name, factory, n_clusters, max_epochs in configs:
        X = factory()
        X_train, X_val, X_test = held_out_split(X, seed=0)
        for seed in seeds:
            reference_centres, reference_seconds = lloyd_at_25(
                X_train, n_clusters, seed
            )
            ref_val = mse(X_val, reference_centres)
            ref_test = mse(X_test, reference_centres)
            rows.append(
                {
                    "dataset": name,
                    "n_train": X_train.shape[0],
                    "n_clusters": n_clusters,
                    "n_features": X_train.shape[1],
                    "seed": seed,
                    "method": "lloyd@25",
                    "epochs": 25,
                    "val_mse": ref_val,
                    "test_mse": ref_test,
                    "rel_test_mse": 0.0,
                    "seconds": reference_seconds,
                }
            )

            for epochs in range(1, max_epochs + 1):
                centres, seconds, _ = emeans_at(X_train, n_clusters, epochs, seed)
                val = mse(X_val, centres)
                test = mse(X_test, centres)
                rows.append(
                    {
                        "dataset": name,
                        "n_train": X_train.shape[0],
                        "n_clusters": n_clusters,
                        "n_features": X_train.shape[1],
                        "seed": seed,
                        "method": "emeans",
                        "epochs": epochs,
                        "val_mse": val,
                        "test_mse": test,
                        "rel_test_mse": relative_mse(test, ref_test),
                        "seconds": seconds,
                    }
                )

            budgets = range(1, max_epochs + 1) if minibatch_per_budget else [max_epochs]
            for budget in budgets:
                start = time.perf_counter()
                _, batch_size, ratio, centres = minibatch_sweep(
                    X_train,
                    X_val,
                    n_clusters,
                    seed,
                    max_iter=budget,
                    batch_sizes=batch_sizes,
                )
                seconds = time.perf_counter() - start
                test = mse(X_test, centres)
                rows.append(
                    {
                        "dataset": name,
                        "n_train": X_train.shape[0],
                        "n_clusters": n_clusters,
                        "n_features": X_train.shape[1],
                        "seed": seed,
                        "method": f"minibatch({batch_size},{ratio})",
                        "epochs": budget,
                        "val_mse": mse(X_val, centres),
                        "test_mse": test,
                        "rel_test_mse": relative_mse(test, ref_test),
                        "seconds": seconds,
                    }
                )
    RESULTS.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarise(rows):
    """Print epochs-to-reference and the best relative test MSE per run."""
    for name in sorted({row["dataset"] for row in rows}):
        for seed in sorted({row["seed"] for row in rows if row["dataset"] == name}):
            runs = [r for r in rows if r["dataset"] == name and r["seed"] == seed]
            reference = next(r for r in runs if r["method"] == "lloyd@25")
            emeans_runs = sorted(
                (r for r in runs if r["method"] == "emeans"),
                key=lambda r: r["epochs"],
            )
            reached = next(
                (
                    r["epochs"]
                    for r in emeans_runs
                    if r["val_mse"] <= reference["val_mse"]
                ),
                None,
            )
            best = min(r["rel_test_mse"] for r in emeans_runs)
            print(
                f"{name:16s} seed={seed} epochs_to_ref={reached} "
                f"best_rel_test_mse={best:+.4f}"
            )


def main():
    warnings.simplefilter("ignore", UserWarning)  # the D-021 range notice
    mode = sys.argv[1] if len(sys.argv) > 1 else "suite"
    if mode not in MODES:
        raise SystemExit(f"mode must be one of {sorted(MODES)}, got {mode!r}")
    configs, seeds, stem, per_budget = MODES[mode]
    out = RESULTS / f"{stem}.csv"
    rows = run(
        configs,
        seeds,
        out,
        batch_sizes=SWEEP_BATCH_SIZES[mode],
        minibatch_per_budget=per_budget,
    )
    summarise(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
