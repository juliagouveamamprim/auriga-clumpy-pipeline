import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "cuts"
    / "scripts"
    / "scan_discarded_js_distribution.py"
)

spec = importlib.util.spec_from_file_location(
    "scan_discarded_js_distribution",
    SCRIPT_PATH,
)
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def write_jsmax_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=scan.JSMAX_CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def jsmax_row(scenario, repop_id, js_max):
    return {
        "scenario": scenario,
        "repop_id": repop_id,
        "n_saved": 10,
        "js_max": js_max,
    }


def test_jsmax_lookup_matches_exact_catalogue_key(tmp_path):
    path = tmp_path / "jsmax.csv"
    write_jsmax_csv(
        path,
        [
            jsmax_row("resilient", 1, 20.0),
            jsmax_row("fragile", 0, 10.0),
        ],
    )

    lookup = scan.load_jsmax_lookup(
        path,
        {("fragile", 0), ("resilient", 1)},
    )

    assert lookup[("fragile", 0)] == scan.JsMaxRecord(10, 10.0)
    assert lookup[("resilient", 1)] == scan.JsMaxRecord(10, 20.0)


def test_jsmax_lookup_rejects_missing_catalogue(tmp_path):
    path = tmp_path / "jsmax.csv"
    write_jsmax_csv(path, [jsmax_row("fragile", 0, 10.0)])

    with pytest.raises(ValueError, match="Missing Js-max entry"):
        scan.load_jsmax_lookup(path, {("resilient", 0)})


def test_jsmax_lookup_rejects_duplicate_catalogue(tmp_path):
    path = tmp_path / "jsmax.csv"
    write_jsmax_csv(
        path,
        [
            jsmax_row("fragile", 0, 10.0),
            jsmax_row("fragile", 0, 11.0),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate Js-max entry"):
        scan.load_jsmax_lookup(path, {("fragile", 0)})


@pytest.mark.parametrize("js_max", ["nan", "inf", "0", "-1"])
def test_jsmax_lookup_rejects_invalid_maximum(tmp_path, js_max):
    path = tmp_path / "jsmax.csv"
    write_jsmax_csv(path, [jsmax_row("fragile", 0, js_max)])

    with pytest.raises(ValueError, match="Invalid js_max"):
        scan.load_jsmax_lookup(path, {("fragile", 0)})


def test_classification_and_discard_selection(monkeypatch):
    columns = {
        name: index
        for index, name in enumerate(scan.REQUIRED_H5_COLUMNS)
    }
    array = np.array(
        [
            [1.0, 10.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0],
            [10.0, 10.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0],
            [20.0, 10.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [30.0, 10.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [np.nan, 10.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    monkeypatch.setattr(
        scan,
        "clumpy_central_pixel_proxy_from_js",
        lambda **_kwargs: np.array([2.0, 8.0]),
    )

    valid, pointlike_discarded, extended_discarded = (
        scan.classify_discarded(
            array=array,
            column_indices=columns,
            theta_min_deg=1.0,
            nside=8,
            j_cut=5.0,
        )
    )

    np.testing.assert_array_equal(valid, [True, True, True, True, False])
    np.testing.assert_array_equal(
        pointlike_discarded,
        [True, False, False, False, False],
    )
    np.testing.assert_array_equal(
        extended_discarded,
        [False, False, True, False, False],
    )


def test_histogram_counts_underflow_overflow_and_invalid():
    result = scan.empty_population(n_bins=2)
    edges = np.array([-2.0, -1.0, 0.0])

    scan.accumulate_discarded(
        result=result,
        js=np.array([0.1, 1.0, 10.0, 100.0, 1000.0, np.nan]),
        js_max_cat=100.0,
        log_relative_edges=edges,
    )

    np.testing.assert_array_equal(result["histogram"], [1, 2])
    assert result["n_discarded"] == 6
    assert result["n_underflow"] == 1
    assert result["n_overflow"] == 1
    assert result["n_invalid_j_rel"] == 1
    assert result["max_js_discarded"] == 1000.0
    assert result["max_j_rel_discarded"] == 10.0


def test_process_catalogue_bins_each_discarded_population(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "catalogue.h5"
    data = np.array(
        [
            [1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [20.0, 10.0, 8.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [30.0, 10.0, 8.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [np.nan, 10.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("iteration_0/data", data=data)
        dataset.attrs["column_names"] = scan.REQUIRED_H5_COLUMNS

    monkeypatch.setattr(
        scan,
        "compute_pixel_reference",
        lambda **_kwargs: {"j_pixel_ref": 10.0},
    )
    monkeypatch.setattr(
        scan,
        "clumpy_central_pixel_proxy_from_js",
        lambda **_kwargs: np.array([2.0, 8.0]),
    )

    result = scan.process_catalogue(
        path=path,
        js_max_cat=100.0,
        expected_n_saved=5,
        nside=8,
        f_value=0.5,
        chunk_size=10,
        log_relative_edges=np.array([-3.0, -1.0, 0.0]),
    )

    np.testing.assert_array_equal(result["hist_pointlike"], [1, 0])
    np.testing.assert_array_equal(result["hist_extended"], [0, 1])
    np.testing.assert_array_equal(result["hist_all"], [1, 1])
    assert result["n_discarded_pointlike"] == 1
    assert result["n_discarded_extended"] == 1
    assert result["n_discarded_all"] == 2
    assert result["max_js_discarded_all"] == 20.0
    assert result["max_j_rel_discarded_all"] == 0.2
    assert result["n_input_rows"] == 5
    assert result["n_invalid_input_rows"] == 1


def test_process_catalogue_rejects_jsmax_row_count_mismatch(tmp_path):
    path = tmp_path / "catalogue.h5"
    data = np.ones((3, len(scan.REQUIRED_H5_COLUMNS)))
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("iteration_0/data", data=data)
        dataset.attrs["column_names"] = scan.REQUIRED_H5_COLUMNS

    with pytest.raises(ValueError, match=r"CSV n_saved=2, HDF5 rows=3"):
        scan.process_catalogue(
            path=path,
            js_max_cat=100.0,
            expected_n_saved=2,
            nside=8,
            f_value=1.0e-3,
            chunk_size=2,
            log_relative_edges=np.array([-2.0, -1.0, 0.0]),
        )


def synthetic_catalogue_result(n_bins):
    result = {
        "j_pixel_ref": 10.0,
        "n_input_rows": 4,
        "n_valid_rows": 4,
        "n_invalid_input_rows": 0,
    }
    for population in scan.POPULATIONS:
        result[f"hist_{population}"] = np.ones(n_bins, dtype=np.int64)
        result[f"n_discarded_{population}"] = n_bins
        result[f"n_underflow_{population}"] = 0
        result[f"n_overflow_{population}"] = 0
        result[f"n_invalid_j_rel_{population}"] = 0
        result[f"max_js_discarded_{population}"] = 1.0
        result[f"max_j_rel_discarded_{population}"] = 0.1
    return result


def test_checkpoint_resume_skips_completed_catalogues(tmp_path):
    input_root = tmp_path / "inputs"
    jsmax_csv = tmp_path / "jsmax.csv"
    output_npz = tmp_path / "checkpoint.npz"
    repop_ids = np.array([0, 1], dtype=np.int64)
    edges = np.array([-2.0, -1.0, 0.0])

    for repop_id in repop_ids:
        path = scan.catalogue_path(input_root, int(repop_id), "fragile")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    args = SimpleNamespace(
        input_root=input_root,
        jsmax_csv=jsmax_csv,
        output_npz=output_npz,
        scenarios=("fragile",),
        nside=8,
        f=1.0e-3,
        chunk_size=2,
    )
    lookup = {
        ("fragile", 0): scan.JsMaxRecord(4, 10.0),
        ("fragile", 1): scan.JsMaxRecord(4, 20.0),
    }
    first_calls = []

    def interrupted_process(**kwargs):
        first_calls.append(
            (
                kwargs["path"].parent.name,
                kwargs["js_max_cat"],
                kwargs["expected_n_saved"],
            )
        )
        if len(first_calls) == 2:
            raise RuntimeError("interrupted")
        return synthetic_catalogue_result(n_bins=2)

    with pytest.raises(RuntimeError, match="interrupted"):
        scan.run_catalogues(
            args=args,
            repop_ids=repop_ids,
            jsmax_lookup=lookup,
            log_relative_edges=edges,
            process_catalogue_fn=interrupted_process,
        )

    assert first_calls == [
        ("repop_0000", 10.0, 4),
        ("repop_0001", 20.0, 4),
    ]

    metadata = scan.checkpoint_metadata(args, repop_ids, edges)
    checkpoint_entries = scan.load_checkpoint(output_npz, metadata)
    assert len(checkpoint_entries) == 1
    assert checkpoint_entries[0]["repop_id"] == 0
    assert checkpoint_entries[0]["jsmax_n_saved"] == 4

    resumed_calls = []

    def resumed_process(**kwargs):
        resumed_calls.append(
            (
                kwargs["path"],
                kwargs["js_max_cat"],
                kwargs["expected_n_saved"],
            )
        )
        return synthetic_catalogue_result(n_bins=2)

    entries = scan.run_catalogues(
        args=args,
        repop_ids=repop_ids,
        jsmax_lookup=lookup,
        log_relative_edges=edges,
        process_catalogue_fn=resumed_process,
    )

    assert len(entries) == 2
    assert resumed_calls == [
        (
            scan.catalogue_path(input_root, 1, "fragile").resolve(),
            20.0,
            4,
        )
    ]
