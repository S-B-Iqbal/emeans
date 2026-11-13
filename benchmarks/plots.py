# SPDX-License-Identifier: BSD-3-Clause
"""Plots for the benchmark report (Phase 4, task P4.3).

Reads ``benchmarks/results/suite.csv`` and writes PNGs into ``benchmarks/plots/``.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"


def load(path=None):
    """Read the CSV written by ``run_suite.py``."""
    path = RESULTS / "suite.csv" if path is None else path
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def epochs_to_reference(rows):
    """Smallest epoch budget reaching the Lloyd@25 validation MSE."""
    out = defaultdict(list)
    datasets = sorted({row["dataset"] for row in rows})
    for name in datasets:
        seeds = sorted({row["seed"] for row in rows if row["dataset"] == name})
        for seed in seeds:
            runs = [r for r in rows if r["dataset"] == name and r["seed"] == seed]
            reference = next(r for r in runs if r["method"] == "lloyd@25")
            emeans_runs = sorted(
                (r for r in runs if r["method"] == "emeans"),
                key=lambda r: int(r["epochs"]),
            )
            reached = next(
                (
                    int(r["epochs"])
                    for r in emeans_runs
                    if float(r["val_mse"]) <= float(reference["val_mse"])
                ),
                None,
            )
            if reached is not None:
                out[name].append(reached)
    return out


def plot_epochs(rows, out_dir):
    """Bar chart of the median epochs needed to match Lloyd@25."""
    reached = epochs_to_reference(rows)
    if not reached:
        return None
    names = sorted(reached)
    medians = [statistics.median(reached[name]) for name in names]
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.bar(names, medians, color="#4c72b0")
    axes.axhline(25, color="#c44e52", linestyle="--", label="Lloyd@25")
    axes.set_ylabel("epochs to reach Lloyd@25 (validation)")
    axes.set_title("e-means reaches Lloyd@25 MSE in a few epochs (median of seeds)")
    axes.legend()
    figure.tight_layout()
    path = out_dir / "epochs_to_reference.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _series(rows, dataset, method_prefix):
    """(epochs, median relative test MSE, median seconds) per budget for one method."""
    points = defaultdict(list)
    for row in rows:
        if row["dataset"] != dataset:
            continue
        if not row["method"].startswith(method_prefix):
            continue
        points[int(row["epochs"])].append(
            (float(row["rel_test_mse"]), float(row["seconds"]))
        )
    epochs, rel, seconds = [], [], []
    for value in sorted(points):
        epochs.append(value)
        rel.append(statistics.median(p[0] for p in points[value]))
        seconds.append(statistics.median(p[1] for p in points[value]))
    return epochs, rel, seconds


def plot_vs_epochs(rows, out_dir, datasets=None):
    """Relative test MSE against the epoch budget, one line per dataset."""
    datasets = sorted({r["dataset"] for r in rows}) if datasets is None else datasets
    figure, axes = plt.subplots(figsize=(7, 4))
    for name in datasets:
        epochs, rel, _ = _series(rows, name, "emeans")
        if epochs:
            axes.plot(epochs, [100 * v for v in rel], marker="o", label=name)
    axes.axhline(0.0, color="#c44e52", linestyle="--", label="Lloyd@25")
    axes.set_xlabel("epochs")
    axes.set_ylabel("test MSE relative to Lloyd@25 (%)")
    axes.set_title("Test MSE vs epoch budget (median of 5 seeds)")
    axes.legend(fontsize="small")
    figure.tight_layout()
    path = out_dir / "test_mse_vs_epochs.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_vs_time(rows, out_dir):
    """Relative test MSE against wall-clock, EMeans vs MiniBatchKMeans."""
    figure, axes = plt.subplots(figsize=(7, 4))
    for name in sorted({r["dataset"] for r in rows}):
        epochs, rel, seconds = _series(rows, name, "emeans")
        if epochs:
            axes.plot(
                seconds, [100 * v for v in rel], marker="o", label=f"emeans {name}"
            )
    axes.axhline(0.0, color="#c44e52", linestyle="--", label="Lloyd@25")
    axes.set_xscale("log")
    axes.set_xlabel("wall-clock (s, log scale)")
    axes.set_ylabel("test MSE relative to Lloyd@25 (%)")
    axes.set_title("Test MSE vs wall-clock (median of seeds)")
    axes.legend(fontsize="small")
    figure.tight_layout()
    path = out_dir / "test_mse_vs_time.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main():
    rows = load()
    PLOTS.mkdir(parents=True, exist_ok=True)
    for path in (
        plot_epochs(rows, PLOTS),
        plot_vs_epochs(rows, PLOTS),
        plot_vs_time(rows, PLOTS),
    ):
        print("wrote", path)


if __name__ == "__main__":
    main()
