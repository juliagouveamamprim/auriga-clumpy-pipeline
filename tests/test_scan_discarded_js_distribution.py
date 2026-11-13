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


def test_dark_selection_is_inclusive_and_uses_full_validity_mask():
    columns = {
        name: index
        for index, name in enumerate(scan.REQUIRED_H5_COLUMNS + ("Vmax",))
    }
    array = np.array(
        [
            [10.0, 10.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 8.0],
            [20.0, 10.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 8.1],
            [30.0, np.nan, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 7.0],
        ]
    )

    physical_valid, selected, n_excluded = scan.eligible_mask(
        array, columns, max_vmax_km_s=8.0
    )

    np.testing.assert_array_equal(physical_valid, [True, True, False])
    np.testing.assert_array_equal(selected, [True, False, False])
    assert n_excluded == 1


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
            [1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 8.0],
            [10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 8.0],
            [20.0, 10.0, 8.0, 1.0, 1.0, 1.0, 0.0, 0.0, 8.0],
            [30.0, 10.0, 8.0, 1.0, 1.0, 1.0, 0.0, 0.0, 8.1],
            [np.nan, 10.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 7.0],
        ]
    )
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("iteration_0/data", data=data)
        dataset.attrs["column_names"] = scan.REQUIRED_H5_COLUMNS + ("Vmax",)

    reference_calls = []
    monkeypatch.setattr(
        scan,
        "compute_pixel_reference",
        lambda **kwargs: (
            reference_calls.append(kwargs) or {"j_pixel_ref": 10.0}
        ),
    )
    monkeypatch.setattr(
        scan,
        "clumpy_central_pixel_proxy_from_js",
        lambda js, **_kwargs: np.full(len(js), 2.0),
    )

    result = scan.process_catalogue(
        path=path,
        nside=8,
        f_value=0.5,
        chunk_size=10,
        log_relative_edges=np.array([-3.0, -1.0, 0.0]),
        max_vmax_km_s=8.0,
    )

    np.testing.assert_array_equal(result["hist_pointlike"], [1, 0])
    np.testing.assert_array_equal(result["hist_extended"], [0, 1])
    np.testing.assert_array_equal(result["hist_all"], [1, 1])
    assert result["n_discarded_pointlike"] == 1
    assert result["n_discarded_extended"] == 1
    assert result["n_discarded_all"] == 2
    assert result["js_max_cat"] == 20.0
    assert result["max_js_discarded_all"] == 20.0
    assert result["max_j_rel_discarded_all"] == 1.0
    assert result["n_input_rows"] == 5
    assert result["n_invalid_input_rows"] == 1
    assert result["n_physical_valid_rows"] == 4
    assert result["n_valid_rows"] == 3
    assert result["n_vmax_excluded"] == 1
    assert reference_calls[0]["max_vmax_km_s"] == 8.0


def synthetic_catalogue_result(n_bins):
    result = {
        "js_max_cat": 10.0,
        "j_pixel_ref": 10.0,
        "n_input_rows": 4,
        "n_valid_rows": 4,
        "n_physical_valid_rows": 4,
        "n_invalid_input_rows": 0,
        "n_vmax_excluded": 0,
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
    output_npz = tmp_path / "checkpoint.npz"
    repop_ids = np.array([0, 1], dtype=np.int64)
    edges = np.array([-2.0, -1.0, 0.0])

    for repop_id in repop_ids:
        path = scan.catalogue_path(input_root, int(repop_id), "fragile")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    args = SimpleNamespace(
        input_root=input_root,
        output_npz=output_npz,
        scenarios=("fragile",),
        nside=8,
        f=1.0e-3,
        chunk_size=2,
        max_vmax_km_s=8.0,
    )
    first_calls = []

    def interrupted_process(**kwargs):
        first_calls.append(
            (
                kwargs["path"].parent.name,
                kwargs["max_vmax_km_s"],
            )
        )
        if len(first_calls) == 2:
            raise RuntimeError("interrupted")
        return synthetic_catalogue_result(n_bins=2)

    with pytest.raises(RuntimeError, match="interrupted"):
        scan.run_catalogues(
            args=args,
            repop_ids=repop_ids,
            log_relative_edges=edges,
            process_catalogue_fn=interrupted_process,
        )

    assert first_calls == [
        ("repop_0000", 8.0),
        ("repop_0001", 8.0),
    ]

    metadata = scan.checkpoint_metadata(args, repop_ids, edges)
    checkpoint_entries = scan.load_checkpoint(output_npz, metadata)
    assert len(checkpoint_entries) == 1
    assert checkpoint_entries[0]["repop_id"] == 0

    resumed_calls = []

    def resumed_process(**kwargs):
        resumed_calls.append(
            (
                kwargs["path"],
                kwargs["max_vmax_km_s"],
            )
        )
        return synthetic_catalogue_result(n_bins=2)

    entries = scan.run_catalogues(
        args=args,
        repop_ids=repop_ids,
        log_relative_edges=edges,
        process_catalogue_fn=resumed_process,
    )

    assert len(entries) == 2
    assert resumed_calls == [
        (
            scan.catalogue_path(input_root, 1, "fragile").resolve(),
            8.0,
        )
    ]


def test_checkpoint_rejects_different_vmax_selection(tmp_path):
    args = SimpleNamespace(
        input_root=tmp_path / "inputs",
        output_npz=tmp_path / "checkpoint.npz",
        scenarios=("fragile",),
        nside=8,
        f=1.0e-3,
        chunk_size=2,
        max_vmax_km_s=8.0,
    )
    repop_ids = np.array([0], dtype=np.int64)
    edges = np.array([-2.0, -1.0, 0.0])
    metadata = scan.checkpoint_metadata(args, repop_ids, edges)
    entry = {
        "repop_id": 0,
        "scenario": "fragile",
        "source_h5": "catalogue.h5",
        **synthetic_catalogue_result(2),
    }
    scan.save_checkpoint(args.output_npz, metadata, [entry])
    with np.load(args.output_npz, allow_pickle=False) as saved:
        assert str(saved["population_selection"]) == "dark_vmax_le_8kms"
        assert float(saved["max_vmax_km_s"]) == 8.0

    args.max_vmax_km_s = 7.0
    incompatible = scan.checkpoint_metadata(args, repop_ids, edges)
    with pytest.raises(ValueError, match="incompatible"):
        scan.load_checkpoint(args.output_npz, incompatible)


def test_maximum_discarded_summary_returns_repopulation_ids():
    entries = []
    for scenario, repop_id, pointlike, extended in (
        ("fragile", 0, 0.1, 0.4),
        ("fragile", 1, 0.3, 0.2),
        ("resilient", 0, 0.5, 0.6),
    ):
        entry = {
            "scenario": scenario,
            "repop_id": repop_id,
            "js_max_cat": 100.0,
            "max_j_rel_discarded_pointlike": pointlike,
            "max_js_discarded_pointlike": pointlike * 100.0,
            "max_j_rel_discarded_extended": extended,
            "max_js_discarded_extended": extended * 100.0,
        }
        entries.append(entry)

    summary = scan.maximum_discarded_by_scenario(entries)
    lookup = {(row["scenario"], row["population"]): row for row in summary}
    assert lookup[("fragile", "pointlike")]["repop_id"] == 1
    assert lookup[("fragile", "extended")]["repop_id"] == 0
    assert lookup[("resilient", "pointlike")]["repop_id"] == 0
