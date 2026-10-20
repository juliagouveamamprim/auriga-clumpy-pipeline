from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "diagnostics" / "corrections" / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

import consolidate_correction_diagnostics as consolidator  # noqa: E402


SCENARIO = "fragile"
NSIDE = 8
NAMES = ["hydro_res_00001", "hydro_res_00002"]
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
UNITS = [
    "GeV^2 cm^-5",
    "kpc",
    "km s^-1",
    "deg",
    "",
    "kpc",
    "Msun kpc^-3",
    "kpc",
    "kpc",
    "kpc",
]


def case_paths(tmp_path: Path, repop_id: int = 0):
    catalog_root = tmp_path / "catalogues"
    clumpy_root = tmp_path / "clumpy"
    paths = consolidator.analysis.build_case_paths(
        catalog_root, clumpy_root, SCENARIO, repop_id, NSIDE
    )
    return catalog_root, clumpy_root, paths


def write_catalog(
    path: Path,
    *,
    names: list[str] = NAMES,
    js: list[float] | tuple[float, ...] = (100.0, 400.0),
    theta: list[float] | tuple[float, ...] = (0.2, 0.4),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones((len(names), len(COLUMNS)), dtype=np.float64)
    data[:, COLUMNS.index("Js")] = js
    data[:, COLUMNS.index("theta_s")] = theta
    with h5py.File(path, "w") as handle:
        group = handle.create_group("iteration_0")
        dataset = group.create_dataset("data", data=data)
        dataset.attrs["column_names"] = np.asarray(
            COLUMNS, dtype=h5py.string_dtype("utf-8")
        )
        dataset.attrs["units"] = np.asarray(UNITS, dtype=h5py.string_dtype("utf-8"))
        group.create_dataset(
            "halo_name",
            data=np.asarray(names, dtype=h5py.string_dtype("utf-8")),
        )
        group.attrs["n_saved"] = len(names)


def write_list(
    path: Path,
    *,
    names: list[str] = NAMES,
    types: list[str] | None = None,
    rhos: list[float] | tuple[float, ...] = (3.0, 5.0),
    longitude_offset: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    types = types or ["DSPH"] * len(names)
    lines = ["# Name Type l b d z Rdelta rhos rs prof #1 #2 #3"]
    for index, (name, halo_type, rho) in enumerate(zip(names, types, rhos), start=1):
        lines.append(
            f"{name} {halo_type} {index + longitude_offset:.6f} 0.000000 "
            f"10.000000 -1 1.000000 {rho:.8e} 1.000000 kZHAO 1 3 1"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_log(
    path: Path,
    *,
    names: list[str] = NAMES,
    types: list[str] | None = None,
    rendered: list[float] | tuple[float, ...] = (25.0, 1600.0),
    npix: list[int] | tuple[int, ...] = (3, 7),
    indices: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    types = types or ["DSPH"] * len(names)
    indices = indices if indices is not None else list(range(len(names)))
    lines = ["# HALO_INDEX NAME TYPE J_RENDERED NPIX_TOUCHED"]
    for index, name, halo_type, j_value, pixel_count in zip(
        indices, names, types, rendered, npix
    ):
        lines.append(f"{index} {name} {halo_type} {j_value:.16g} {pixel_count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_valid_case(tmp_path: Path, repop_id: int = 0):
    catalog_root, clumpy_root, paths = case_paths(tmp_path, repop_id)
    write_catalog(paths.catalog)
    write_list(paths.raw_list)
    write_list(paths.corrected_list, rhos=(6.0, 2.5))
    write_log(paths.raw_log)
    write_log(
        paths.corrected_log,
        rendered=(100.0, 400.0),
        npix=(5, 4),
    )
    return catalog_root, clumpy_root, paths


def build_one(tmp_path: Path) -> pd.DataFrame:
    catalog_root, clumpy_root, _ = make_valid_case(tmp_path)
    return consolidator.build_repopulation_table(
        catalog_root, clumpy_root, SCENARIO, 0, NSIDE
    )


def test_happy_path_two_repops_exact_schema_values_types_and_order(tmp_path):
    catalog_root, clumpy_root, _ = make_valid_case(tmp_path, 0)
    make_valid_case(tmp_path, 1)

    table = consolidator.consolidate(catalog_root, clumpy_root, SCENARIO, 0, 1, NSIDE)

    assert list(table.columns) == consolidator.FINAL_COLUMNS
    assert list(zip(table["repop_id"], table["halo_index"])) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert pd.api.types.is_integer_dtype(table["repop_id"])
    assert pd.api.types.is_integer_dtype(table["halo_index"])
    assert pd.api.types.is_integer_dtype(table["npix_raw"])
    assert pd.api.types.is_integer_dtype(table["npix_corr"])
    np.testing.assert_allclose(table["theta_s_deg"], [0.2, 0.4] * 2)
    np.testing.assert_allclose(table["J_s"], [100.0, 400.0] * 2)
    np.testing.assert_allclose(table["R_raw"], [0.25, 4.0] * 2)
    np.testing.assert_allclose(table["R_corr"], [1.0, 1.0] * 2)
    np.testing.assert_allclose(table["rho_s_raw"], [12.0, 20.0] * 2)
    np.testing.assert_allclose(table["rho_s_corr"], [24.0, 10.0] * 2)
    np.testing.assert_allclose(table["rho_corr_over_raw"], [2.0, 0.5] * 2)
    np.testing.assert_array_equal(table["npix_raw"], [3, 7] * 2)
    np.testing.assert_array_equal(table["npix_corr"], [5, 4] * 2)


def test_density_ratio_accepts_corrected_list_decimal_serialization(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    raw_rhos = (1.23456789, 9.87654321)
    raw_j = (73.0, 901.0)
    js = (113.0, 557.0)
    corrected = tuple(
        float(f"{rho / np.sqrt(j / target):.8e}")
        for rho, j, target in zip(raw_rhos, raw_j, js)
    )
    write_catalog(paths.catalog, js=js)
    write_list(paths.raw_list, rhos=raw_rhos)
    write_list(paths.corrected_list, rhos=corrected)
    write_log(paths.raw_log, rendered=raw_j)
    write_log(paths.corrected_log, rendered=js, npix=(2, 2))

    table = consolidator.build_repopulation_table(
        catalog_root, clumpy_root, SCENARIO, 0, NSIDE
    )

    np.testing.assert_allclose(table["rho_s_raw"], 4.0 * np.asarray(raw_rhos))
    np.testing.assert_allclose(table["rho_s_corr"], 4.0 * np.asarray(corrected))
    np.testing.assert_allclose(
        table["rho_corr_over_raw"],
        1.0 / np.sqrt(np.asarray(raw_j) / np.asarray(js)),
        rtol=consolidator.RHO_RATIO_RTOL,
        atol=0.0,
    )


def test_cli_defaults_inclusive_range_and_output_name():
    args = consolidator.parse_args([SCENARIO])
    assert args.start_id == 0
    assert args.end_id == 21
    assert args.nside == 2048
    assert consolidator.default_output_path(SCENARIO, 0, 21).name == (
        "corrections_fragile_repop0000_0021.csv"
    )


def test_cli_explicit_output_is_preserved(tmp_path):
    output = tmp_path / "chosen.csv"
    args = consolidator.parse_args(
        [SCENARIO, "--start-id", "3", "--end-id", "4", "--output", str(output)]
    )
    assert args.output == output
    assert list(range(args.start_id, args.end_id + 1)) == [3, 4]


@pytest.mark.parametrize(
    "start_id,end_id,nside,match",
    [
        (2, 1, 8, "start-id"),
        (-1, 1, 8, "non-negative"),
        (0, 1, 3, "power of two"),
    ],
)
def test_invalid_cli_values_are_rejected(start_id, end_id, nside, match):
    args = consolidator.parse_args(
        [
            SCENARIO,
            "--start-id",
            str(start_id),
            "--end-id",
            str(end_id),
            "--nside",
            str(nside),
        ]
    )
    with pytest.raises(ValueError, match=match):
        consolidator.validate_cli_args(args)


@pytest.mark.parametrize(
    "target",
    ["catalog", "raw_list", "corrected_list", "raw_log", "corrected_log"],
)
def test_missing_required_input_fails_fast(tmp_path, target):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    getattr(paths, target).unlink()
    with pytest.raises(consolidator.ConsolidationError, match="missing"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


@pytest.mark.parametrize(
    "target",
    ["catalog", "raw_list", "corrected_list", "raw_log", "corrected_log"],
)
def test_empty_required_input_fails_fast(tmp_path, target):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    getattr(paths, target).write_bytes(b"")
    with pytest.raises(consolidator.ConsolidationError, match="empty"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_duplicate_halo_name_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_list(paths.raw_list, names=[NAMES[0], NAMES[0]])
    with pytest.raises(RuntimeError, match="duplicate halo names"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_reordered_corrected_names_are_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    reversed_names = list(reversed(NAMES))
    write_list(paths.corrected_list, names=reversed_names, rhos=(2.5, 6.0))
    write_log(
        paths.corrected_log,
        names=reversed_names,
        rendered=(400.0, 100.0),
        npix=(4, 5),
    )
    with pytest.raises(consolidator.ConsolidationError, match="halo_name"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_type_divergence_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_list(paths.corrected_list, types=["DSPH", "GAL"], rhos=(6.0, 2.5))
    write_log(
        paths.corrected_log,
        types=["DSPH", "GAL"],
        rendered=(100.0, 400.0),
        npix=(5, 4),
    )
    with pytest.raises(consolidator.ConsolidationError, match="type"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_non_zero_based_halo_index_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_log(paths.raw_log, indices=[1, 2])
    with pytest.raises(RuntimeError, match="zero-based"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_partial_log_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_log(
        paths.corrected_log,
        names=[NAMES[0]],
        rendered=(100.0,),
        npix=(5,),
    )
    with pytest.raises(consolidator.ConsolidationError, match="row-count mismatch"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_non_rhos_list_field_change_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_list(paths.corrected_list, rhos=(6.0, 2.5), longitude_offset=0.1)
    with pytest.raises(consolidator.ConsolidationError, match="l_deg"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_hdf5_stored_name_mismatch_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_catalog(paths.catalog, names=["wrong_name", NAMES[1]])
    with pytest.raises(consolidator.ConsolidationError, match="identity mismatch"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_noncanonical_halo_name_is_rejected_before_hdf5_read(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    bad_names = ["not_canonical", NAMES[1]]
    write_catalog(paths.catalog, names=bad_names)
    write_list(paths.raw_list, names=bad_names)
    write_list(paths.corrected_list, names=bad_names, rhos=(6.0, 2.5))
    write_log(paths.raw_log, names=bad_names)
    write_log(
        paths.corrected_log,
        names=bad_names,
        rendered=(100.0, 400.0),
        npix=(5, 4),
    )
    with pytest.raises(consolidator.ConsolidationError, match="Non-canonical"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
@pytest.mark.parametrize("source", ["J_s", "J_raw", "J_corr"])
def test_invalid_j_values_are_rejected(tmp_path, source, bad):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    if source == "J_s":
        write_catalog(paths.catalog, js=(bad, 400.0))
    elif source == "J_raw":
        write_log(paths.raw_log, rendered=(bad, 1600.0))
    else:
        write_log(
            paths.corrected_log,
            rendered=(bad, 400.0),
            npix=(5, 4),
        )
    with pytest.raises((consolidator.ConsolidationError, RuntimeError), match=source):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
@pytest.mark.parametrize("stage", ["raw", "corr"])
def test_invalid_density_is_rejected(tmp_path, stage, bad):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    if stage == "raw":
        write_list(paths.raw_list, rhos=(bad, 5.0))
    else:
        write_list(paths.corrected_list, rhos=(bad, 2.5))
    with pytest.raises(
        (consolidator.ConsolidationError, RuntimeError), match="rho|rhos"
    ):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_theta_is_rejected(tmp_path, bad):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_catalog(paths.catalog, theta=(bad, 0.4))
    with pytest.raises(consolidator.ConsolidationError, match="theta_s_deg"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


@pytest.mark.parametrize("bad", [-1, 0])
@pytest.mark.parametrize("stage", ["raw", "corr"])
def test_invalid_npix_is_rejected(tmp_path, stage, bad):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    if stage == "raw":
        write_log(paths.raw_log, npix=(bad, 7))
    else:
        write_log(
            paths.corrected_log,
            rendered=(100.0, 400.0),
            npix=(bad, 4),
        )
    with pytest.raises(consolidator.ConsolidationError, match=f"npix_{stage}"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_inconsistent_corrected_density_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_list(paths.corrected_list, rhos=(6.1, 2.5))
    with pytest.raises(consolidator.ConsolidationError, match="1/sqrt"):
        consolidator.build_repopulation_table(
            catalog_root, clumpy_root, SCENARIO, 0, NSIDE
        )


def test_corrected_log_produces_r_corr_from_j_corr(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_log(
        paths.corrected_log,
        rendered=(90.0, 440.0),
        npix=(5, 4),
    )
    table = consolidator.build_repopulation_table(
        catalog_root, clumpy_root, SCENARIO, 0, NSIDE
    )
    np.testing.assert_allclose(table["R_corr"], [0.9, 1.1])


def test_global_validation_rejects_missing_repop():
    table = pd.DataFrame(
        [[SCENARIO, 0, 0, NAMES[0], 0.2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        columns=consolidator.FINAL_COLUMNS,
    )
    with pytest.raises(consolidator.ConsolidationError, match="coverage"):
        consolidator.validate_consolidated_table(table, SCENARIO, 0, 1)


def test_global_validation_rejects_mixed_scenario(tmp_path):
    table = build_one(tmp_path)
    table.loc[1, "scenario"] = "resilient"
    with pytest.raises(consolidator.ConsolidationError, match="mixing"):
        consolidator.validate_consolidated_table(table, SCENARIO, 0, 0)


@pytest.mark.parametrize("column", ["halo_index", "halo_name"])
def test_global_validation_rejects_duplicate_keys(tmp_path, column):
    table = build_one(tmp_path)
    table.loc[1, column] = table.loc[0, column]
    with pytest.raises(consolidator.ConsolidationError, match="Duplicate final key"):
        consolidator.validate_consolidated_table(table, SCENARIO, 0, 0)


def test_main_failure_does_not_create_partial_output(tmp_path):
    catalog_root, clumpy_root, _ = make_valid_case(tmp_path, 0)
    output = tmp_path / "result.csv"
    exit_code = consolidator.main(
        [
            SCENARIO,
            "--start-id",
            "0",
            "--end-id",
            "1",
            "--nside",
            str(NSIDE),
            "--catalog-root",
            str(catalog_root),
            "--clumpy-root",
            str(clumpy_root),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 1
    assert not output.exists()


def test_main_writes_one_consolidated_csv_for_inclusive_range(tmp_path):
    catalog_root, clumpy_root, _ = make_valid_case(tmp_path, 0)
    make_valid_case(tmp_path, 1)
    output = tmp_path / "result.csv"
    exit_code = consolidator.main(
        [
            SCENARIO,
            "--start-id",
            "0",
            "--end-id",
            "1",
            "--nside",
            str(NSIDE),
            "--catalog-root",
            str(catalog_root),
            "--clumpy-root",
            str(clumpy_root),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    loaded = pd.read_csv(output)
    assert list(loaded.columns) == consolidator.FINAL_COLUMNS
    assert list(loaded["repop_id"]) == [0, 0, 1, 1]


def test_total_fits_is_not_required(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    assert not paths.total_fits.exists()
    table = consolidator.build_repopulation_table(
        catalog_root, clumpy_root, SCENARIO, 0, NSIDE
    )
    assert len(table) == 2


def test_sparse_reader_uses_only_selected_rows_and_columns():
    class RecordingDataset:
        def __init__(self, array):
            self.array = np.asarray(array)
            self.keys = []

        def __getitem__(self, key):
            self.keys.append(key)
            components = key if isinstance(key, tuple) else (key,)
            if any(isinstance(component, slice) for component in components):
                raise AssertionError("full/slice dataset read is forbidden")
            return self.array[key]

    data = RecordingDataset(
        [
            [10.0, 100.0, 0.1],
            [20.0, 200.0, 0.2],
            [30.0, 300.0, 0.3],
        ]
    )
    names = RecordingDataset([b"h1", b"h2", b"h3"])
    rows = np.asarray([0, 2], dtype=np.int64)

    selected_names, js, theta = consolidator._read_sparse_hdf5_rows(
        data, names, rows, 0, 2
    )

    np.testing.assert_array_equal(selected_names, [b"h1", b"h3"])
    np.testing.assert_allclose(js, [10.0, 30.0])
    np.testing.assert_allclose(theta, [0.1, 0.3])
    assert len(names.keys) == 1
    assert len(data.keys) == 2
    np.testing.assert_array_equal(names.keys[0], rows)
    np.testing.assert_array_equal(data.keys[0][0], rows)
    assert data.keys[0][1] == 0
    np.testing.assert_array_equal(data.keys[1][0], rows)
    assert data.keys[1][1] == 2


def test_catalog_is_opened_once_per_sparse_load(tmp_path, monkeypatch):
    _, _, paths = make_valid_case(tmp_path)
    original_file = h5py.File
    calls = []

    def counting_file(*args, **kwargs):
        calls.append(args[0])
        return original_file(*args, **kwargs)

    monkeypatch.setattr(consolidator.h5py, "File", counting_file)
    rows, js, theta = consolidator.load_catalog_values_sparse(paths.catalog, NAMES)

    assert calls == [paths.catalog]
    np.testing.assert_array_equal(rows, [0, 1])
    np.testing.assert_allclose(js, [100.0, 400.0])
    np.testing.assert_allclose(theta, [0.2, 0.4])


def test_atomic_csv_round_trips_schema_and_floats(tmp_path):
    table = build_one(tmp_path)
    output = tmp_path / "nested" / "result.csv"
    consolidator.write_csv_atomic(table, output)

    loaded = pd.read_csv(output)
    assert list(loaded.columns) == consolidator.FINAL_COLUMNS
    np.testing.assert_allclose(loaded["J_s"], table["J_s"], rtol=0.0, atol=0.0)
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
