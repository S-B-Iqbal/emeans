# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests over the committed benchmark CSVs (P9.3).

The README, the report and the upstream drafts quote these numbers, so a change that
quietly breaks the mechanism behind them (revival, the cosine schedule, the automatic
laws) must fail here instead of invalidating published evidence.  The tests only read
committed CSVs, so they are fast; a missing CSV skips with an explanation.
"""

import csv
import statistics
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"
LARGE = RESULTS / "large.csv"

EMEANS = "emeans"
REVIVAL_OFF = "emeans(revival_off)"

pytestmark = pytest.mark.skipif(
    not LARGE.exists(),
    reason="results/large.csv missing; run `python benchmarks/run_suite.py large`",
)


@pytest.fixture(scope="module")
def rows():
    with LARGE.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _select(rows, dataset, method, epochs=None):
    return [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["method"] == method
        and (epochs is None or int(row["epochs"]) == epochs)
    ]


def _median(values):
    return statistics.median(values)


def test_in_range_suite_has_the_expected_shape(rows):
    datasets = {row["dataset"] for row in rows}
    assert {"blobs-96d-n100k-k512", "blobs-32d-n100k-k512"} <= datasets
    assert {
        "aniso-96d-n100k-k512",
    } <= datasets  # the P8.6 stress case must keep being reported
    for row in rows:
        assert row["n_train"] == "100000"
        assert row["n_clusters"] == "512"
    assert len(rows) >= 200  # 3 datasets x 5 seeds x (1 + 10 + 5) rows


def test_lloyd_rows_are_the_reference(rows):
    lloyd = [row for row in rows if row["method"] == "lloyd@25"]
    assert lloyd
    assert all(float(row["rel_test_mse"]) == 0.0 for row in lloyd)
    assert all(int(row["epochs"]) == 25 for row in lloyd)


def test_emeans_beats_lloyd_at_five_epochs_with_revival(rows):
    for dataset, floor in [
        ("blobs-96d-n100k-k512", -0.25),
        ("blobs-32d-n100k-k512", -0.10),
        ("aniso-96d-n100k-k512", -0.08),
    ]:
        rel = [
            float(r["rel_test_mse"]) for r in _select(rows, dataset, EMEANS, epochs=5)
        ]
        assert len(rel) == 5, dataset
        assert _median(rel) < floor, (dataset, _median(rel))
        assert max(rel) < 0.0, (dataset, max(rel))  # even the worst seed beats Lloyd@25


def test_revival_is_what_makes_emeans_win(rows):
    for dataset in (
        "blobs-96d-n100k-k512",
        "blobs-32d-n100k-k512",
        "aniso-96d-n100k-k512",
    ):
        on = _select(rows, dataset, EMEANS, epochs=5)
        off = _select(rows, dataset, REVIVAL_OFF, epochs=5)
        assert len(on) == len(off) == 5, dataset

        assert _median([int(r["n_revivals"]) for r in on]) > 50
        assert all(int(r["n_revivals"]) == 0 for r in off)

        gain = _median([float(r["rel_test_mse"]) for r in on])
        without = _median([float(r["rel_test_mse"]) for r in off])
        assert gain < -0.08, (dataset, gain)
        assert without > 0.0, (
            dataset,
            without,
        )  # without revival it does not beat Lloyd@25
        assert without - gain > 0.10, (dataset, gain, without)


def test_one_epoch_reaches_lloyd_on_the_blob_datasets(rows):
    for dataset in ("blobs-96d-n100k-k512", "blobs-32d-n100k-k512"):
        reference = {
            row["seed"]: float(row["val_mse"])
            for row in _select(rows, dataset, "lloyd@25")
        }
        candidates = _select(rows, dataset, EMEANS)
        reached_at_one = 0
        for seed, ref in reference.items():
            first = min(
                (
                    int(row["epochs"])
                    for row in candidates
                    if row["seed"] == seed and float(row["val_mse"]) <= ref
                ),
                default=None,
            )
            reached_at_one += first == 1
        assert reached_at_one >= 4, (dataset, reached_at_one)


def test_the_tuned_minibatch_baseline_is_reported_as_ahead(rows):
    """The documented loss: at equal epochs the validation-tuned baseline wins."""
    for dataset in ("blobs-96d-n100k-k512", "aniso-96d-n100k-k512"):
        emeans = _median(
            [float(r["rel_test_mse"]) for r in _select(rows, dataset, EMEANS, epochs=5)]
        )
        minibatch = _median(
            [
                float(row["rel_test_mse"])
                for row in rows
                if row["dataset"] == dataset
                and row["method"].startswith("minibatch(")
                and int(row["epochs"]) == 5
            ]
        )
        assert minibatch < emeans, (dataset, emeans, minibatch)


def test_the_openmp_run_reproduces_the_serial_result(rows):
    """P9.2: the Linux/OpenMP run must agree with the serial run on every row."""
    other = RESULTS / "large-omp.csv"
    if not other.exists():
        pytest.skip("results/large-omp.csv missing; run the `benchmark` workflow")
    with other.open(newline="") as handle:
        omp = list(csv.DictReader(handle))

    key = lambda row: (row["dataset"], row["seed"], row["method"], row["epochs"])  # noqa: E731
    serial_by_key = {key(row): row for row in rows}
    omp_by_key = {key(row): row for row in omp}
    shared = sorted(set(serial_by_key) & set(omp_by_key))

    assert len(shared) >= 200
    worst = max(
        abs(float(serial_by_key[k]["test_mse"]) - float(omp_by_key[k]["test_mse"]))
        for k in shared
    )
    assert worst < 1e-9, worst
    assert int(omp_by_key[shared[0]]["n_revivals"]) >= 0  # schema is the same
