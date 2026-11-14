# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for the benchmark runner (P8.2a).

The runner is user-facing (``quick`` / ``suite`` / ``large``) and produces the numbers
quoted in the README, the report and the upstream drafts: a silent break would
invalidate published evidence.  The tests below use a tiny configuration and must
stay under a couple of seconds.
"""

import csv
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1]
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import run_suite  # noqa: E402
from datasets import make_blobs  # noqa: E402

# One config, one seed, two epoch budgets: 1 Lloyd + 2 EMeans + 1-2 MiniBatch rows.
TINY = [("tiny", lambda: make_blobs(1000, 8, 16, seed=0), 16, 2)]


def _rows(tmp_path, **kwargs):
    out = tmp_path / "tiny.csv"
    rows = run_suite.run(TINY, [0], out, **kwargs)
    return rows, out


def test_single_sweep_writes_a_valid_csv(tmp_path):
    rows, out = _rows(tmp_path)

    assert len(rows) == 4  # lloyd@25, emeans@1, emeans@2, one MiniBatchKMeans sweep
    with out.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == run_suite.FIELDS
        written = list(reader)
    assert len(written) == len(rows)

    methods = [row["method"] for row in rows]
    assert methods[0] == "lloyd@25"
    assert sum(m.startswith("minibatch(") for m in methods) == 1
    assert rows[0]["rel_test_mse"] == 0.0  # the reference is relative to itself
    assert all(row["n_features"] == 8 for row in rows)


def test_per_budget_sweep_adds_one_row_per_epoch(tmp_path):
    rows, _ = _rows(tmp_path, minibatch_per_budget=True)

    assert len(rows) == 5  # + one MiniBatchKMeans row per epoch budget
    budgets = sorted(
        int(row["epochs"]) for row in rows if row["method"].startswith("minibatch(")
    )
    assert budgets == [1, 2]


def test_modes_and_sweeps_are_consistent():
    assert set(run_suite.MODES) == {"quick", "suite", "large"}
    assert set(run_suite.SWEEP_BATCH_SIZES) == set(run_suite.MODES)
    for _configs, seeds, stem, per_budget in run_suite.MODES.values():
        assert seeds and isinstance(stem, str) and isinstance(per_budget, bool)
    assert (
        run_suite.MODES["large"][3] is True
    )  # the in-range run compares at equal epochs


@pytest.mark.parametrize("mode", ["quick", "suite", "large"])
def test_modes_have_configs_with_expected_shape(mode):
    configs = run_suite.MODES[mode][0]
    assert configs
    for name, factory, n_clusters, max_epochs in configs:
        assert isinstance(name, str) and name
        assert n_clusters >= 2 and max_epochs >= 1
        assert callable(factory)


def test_summarise_accepts_the_rows_it_produces(tmp_path, capsys):
    rows, _ = _rows(tmp_path)

    run_suite.summarise(rows)

    assert "tiny" in capsys.readouterr().out
