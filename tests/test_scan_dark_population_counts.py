import csv
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "catalogue"
    / "scripts"
    / "scan_dark_population_counts.py"
)
spec = importlib.util.spec_from_file_location(
    "scan_dark_population_counts", SCRIPT_PATH
)
scan = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scan
spec.loader.exec_module(scan)


COLUMNS = [
    "Js",
    "D_Earth",
    "Vmax",
    "theta_s",
    "Cv",
    "r_s",
    "rho_s",
    "Xearth",
    "Yearth",
    "Zearth",
]


def write_catalogue(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "iteration_0/data", data=np.asarray(rows, dtype=np.float64)
        )
        dataset.attrs["column_names"] = COLUMNS


def valid_row(vmax, theta_s, *, js=1.0):
    return [js, 10.0, vmax, theta_s, 100.0, 1.0, 1.0e7, 1.0, 2.0, 3.0]


def synthetic_counts():
    return {
        "n_input_rows": 7,
        "n_physical_valid": 6,
        "n_vmax_excluded": 1,
        "n_total": 5,
        "n_pointlike": 3,
        "n_extended": 2,
    }


def test_scan_uses_production_validity_and_pre_cut_classification(tmp_path):
    path = tmp_path / "catalogue.h5"
    rows = [
        valid_row(8.0, 0.029),
        valid_row(7.0, 0.03),
        valid_row(1.0, 0.031),
        valid_row(8.1, 0.001),
        valid_row(2.0, 0.001, js=np.nan),
        valid_row(2.0, 0.0),
    ]
    write_catalogue(path, rows)

    result = scan.scan_catalogue(path, chunk_size=2)

    assert result == {
        "n_input_rows": 6,
        "n_physical_valid": 5,
        "n_vmax_excluded": 1,
        "n_total": 4,
        "n_pointlike": 2,
        "n_extended": 2,
    }


def test_scan_rejects_invalid_vmax_for_otherwise_valid_halo(tmp_path):
    path = tmp_path / "catalogue.h5"
    write_catalogue(path, [valid_row(0.0, 0.01)])

    with pytest.raises(ValueError, match="Invalid Vmax.*row 0"):
        scan.scan_catalogue(path, chunk_size=1)


def make_args(tmp_path, repop_end=1):
    return SimpleNamespace(
        input_root=tmp_path / "input",
        output_csv=tmp_path / "counts.csv",
        repop_start=0,
        repop_end=repop_end,
        scenarios=("fragile",),
        chunk_size=3,
    )


def test_atomic_checkpoint_resumes_without_duplicates(tmp_path):
    args = make_args(tmp_path)
    for repop_id in range(2):
        path = scan.catalogue_path(args.input_root, repop_id, "fragile")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    calls = []

    def interrupted(path, chunk_size):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return synthetic_counts()

    with pytest.raises(RuntimeError, match="interrupted"):
        scan.run_catalogues(args, scan_catalogue_fn=interrupted)

    checkpoint = scan.load_checkpoint(args.output_csv)
    assert list(checkpoint) == [("fragile", 0)]

    resumed_calls = []

    def resumed(path, chunk_size):
        resumed_calls.append(path)
        return synthetic_counts()

    records = scan.run_catalogues(args, scan_catalogue_fn=resumed)
    assert resumed_calls == [
        scan.catalogue_path(args.input_root, 1, "fragile")
    ]
    assert sorted(records) == [("fragile", 0), ("fragile", 1)]
    with args.output_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2


def test_summary_reports_mean_and_aggregate_fractions(tmp_path):
    path = tmp_path / "a.h5"
    rows = [
        scan.build_output_row(
            "fragile",
            0,
            path,
            2,
            {
                "n_input_rows": 10,
                "n_physical_valid": 10,
                "n_vmax_excluded": 0,
                "n_total": 10,
                "n_pointlike": 9,
                "n_extended": 1,
            },
        ),
        scan.build_output_row(
            "fragile",
            1,
            path,
            2,
            {
                "n_input_rows": 2,
                "n_physical_valid": 2,
                "n_vmax_excluded": 0,
                "n_total": 2,
                "n_pointlike": 1,
                "n_extended": 1,
            },
        ),
    ]

    summary = scan.summarize(rows)[0]

    assert summary["mean_n_total"] == 6.0
    assert summary["mean_n_pointlike"] == 5.0
    assert summary["mean_n_extended"] == 1.0
    assert summary["mean_pointlike_fraction"] == pytest.approx(0.7)
    assert summary["mean_extended_fraction"] == pytest.approx(0.3)
    assert summary["aggregate_pointlike_fraction"] == pytest.approx(10 / 12)
    assert summary["aggregate_extended_fraction"] == pytest.approx(2 / 12)


def test_checked_in_figure9_counts_are_deduplicated_and_validate():
    figure9 = (
        REPOSITORY_ROOT
        / "outputs"
        / "diagnostics"
        / "cuts"
        / "figure9"
        / "repops_0000_0049_fragile-resilient_nside2048_"
        "dark_vmax_le_8kms_figure9_combined.csv"
    )
    expected = scan.load_figure9_counts(figure9)
    assert len(expected) == 100
    rows = [
        {
            "scenario": scenario,
            "repop_id": repop_id,
            "n_total": counts[0],
            "n_pointlike": counts[1],
            "n_extended": counts[2],
        }
        for (scenario, repop_id), counts in expected.items()
    ]

    assert scan.validate_against_figure9(rows, figure9) == 100
    rows[0]["n_extended"] += 1
    with pytest.raises(ValueError, match="Figure 9 count mismatch"):
        scan.validate_against_figure9(rows, figure9)


def test_total_count_npz_reports_narrower_mask_differences(tmp_path):
    path = tmp_path / "totals.npz"
    np.savez(
        path,
        max_vmax_km_s=np.asarray(8.0),
        fragile_repop_ids=np.asarray([0, 1]),
        fragile_n_total=np.asarray([5, 7]),
        resilient_repop_ids=np.asarray([0]),
        resilient_n_total=np.asarray([9]),
    )
    rows = [
        {"scenario": "fragile", "repop_id": 0, "n_total": 5},
        {"scenario": "fragile", "repop_id": 1, "n_total": 6},
    ]

    compared, mismatches = scan.compare_total_count_npz(rows, path)

    assert compared == 2
    assert mismatches == [("fragile", 1, 6, 7)]


def test_checked_in_figure9_totals_match_500_repopulation_npz():
    figure9 = (
        REPOSITORY_ROOT
        / "outputs"
        / "diagnostics"
        / "cuts"
        / "figure9"
        / "repops_0000_0049_fragile-resilient_nside2048_"
        "dark_vmax_le_8kms_figure9_combined.csv"
    )
    total_npz = (
        REPOSITORY_ROOT
        / "outputs"
        / "diagnostics"
        / "cuts"
        / "relative_js_cumulative"
        / "relative_js_cumulative_0500repops.npz"
    )
    expected = scan.load_figure9_counts(figure9)
    rows = [
        {
            "scenario": scenario,
            "repop_id": repop_id,
            "n_total": counts[0],
        }
        for (scenario, repop_id), counts in expected.items()
    ]

    compared, mismatches = scan.compare_total_count_npz(rows, total_npz)

    assert compared == 100
    assert mismatches == []
