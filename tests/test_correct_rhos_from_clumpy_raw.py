import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import correct_rhos_from_clumpy_raw as correction  # noqa: E402


NAMES = ["hydro_res_00001", "hydro_res_00002"]


def write_catalog(path, js=(100.0, 400.0), names=NAMES):
    path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0.2, 0.4, len(js), dtype=np.float64)
    data = np.column_stack(
        [np.asarray(js, dtype=np.float64), theta]
    )
    with h5py.File(path, "w") as handle:
        group = handle.create_group("iteration_0")
        dataset = group.create_dataset("data", data=data)
        dataset.attrs["column_names"] = np.asarray(
            ["Js", "theta_s"], dtype=h5py.string_dtype("utf-8")
        )
        dataset.attrs["units"] = np.asarray(
            ["GeV^2 cm^-5", "deg"], dtype=h5py.string_dtype("utf-8")
        )
        group.create_dataset(
            "halo_name",
            data=np.asarray(names, dtype=h5py.string_dtype("utf-8")),
        )
        group.attrs["n_saved"] = len(data)


def write_raw_list(path, names=NAMES, rhos=(3.0, 5.0), distances=(9.876543, 1.2)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, rho, distance in zip(names, rhos, distances):
        rows.append(
            f"{name} DSPH 0 0 {distance} -1 0.123456 {rho} "
            "0.123456 kZHAO 1 3 1\n"
        )
    path.write_text("".join(rows), encoding="utf-8")


def write_raw_log(path, names=NAMES, raw_js=(25.0, 1600.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{index} {name} DSPH {j_raw} 1\n"
            for index, (name, j_raw) in enumerate(zip(names, raw_js))
        ),
        encoding="utf-8",
    )


def test_hdf5_js_is_authoritative_target_and_factor_ignores_list_rounding(tmp_path):
    catalog = tmp_path / "catalog.h5"
    raw_list = tmp_path / "raw.txt"
    raw_log = tmp_path / "raw.log"
    write_catalog(catalog)
    write_raw_list(raw_list, distances=(0.000001, 999999.999999))
    write_raw_log(raw_log)

    list_df = correction.load_clumpy_list(raw_list)
    log_df = correction.load_rendered_log(raw_log)
    rows, catalog_js = correction.load_catalog_js_indexed(catalog, NAMES)
    matched = correction.build_matched_table(
        list_df, log_df, rows, catalog_js
    )

    np.testing.assert_array_equal(matched["hdf5_row"], [0, 1])
    np.testing.assert_allclose(matched["J_catalog"], [100.0, 400.0])
    np.testing.assert_allclose(matched["J_raw"], [25.0, 1600.0])
    np.testing.assert_allclose(matched["R_raw"], [0.25, 4.0])
    np.testing.assert_allclose(matched["correction_factor"], [2.0, 0.5])
    np.testing.assert_allclose(matched["rhos_raw"], [3.0, 5.0])
    np.testing.assert_allclose(matched["rhos_corrected"], [6.0, 2.5])


def test_exact_hdf5_name_mismatch_fails(tmp_path):
    catalog = tmp_path / "catalog.h5"
    write_catalog(catalog, names=["hydro_res_00001", "wrong_name"])
    with pytest.raises(RuntimeError, match="identity mismatch"):
        correction.load_catalog_js_indexed(catalog, NAMES)


def test_missing_hdf5_name_fails_by_inferred_row_check(tmp_path):
    catalog = tmp_path / "catalog.h5"
    write_catalog(catalog, js=(100.0,), names=["hydro_res_00001"])
    with pytest.raises(RuntimeError, match="outside"):
        correction.load_catalog_js_indexed(catalog, NAMES)


def test_duplicate_raw_list_name_fails(tmp_path):
    raw_list = tmp_path / "raw.txt"
    write_raw_list(raw_list, names=[NAMES[0], NAMES[0]])
    with pytest.raises(RuntimeError, match="duplicate halo names"):
        correction.load_clumpy_list(raw_list)


def test_raw_list_log_name_order_mismatch_fails(tmp_path):
    raw_list = tmp_path / "raw.txt"
    raw_log = tmp_path / "raw.log"
    write_raw_list(raw_list)
    write_raw_log(raw_log, names=list(reversed(NAMES)))
    list_df = correction.load_clumpy_list(raw_list)
    log_df = correction.load_rendered_log(raw_log)
    with pytest.raises(RuntimeError, match="names/order"):
        correction.build_matched_table(
            list_df,
            log_df,
            np.asarray([0, 1]),
            np.asarray([100.0, 400.0]),
        )


@pytest.mark.parametrize("bad_js", [0.0, -1.0, np.nan])
def test_invalid_catalog_js_fails(tmp_path, bad_js):
    catalog = tmp_path / "catalog.h5"
    write_catalog(catalog, js=(100.0, bad_js))
    with pytest.raises(RuntimeError, match="finite and positive"):
        correction.load_catalog_js_indexed(catalog, NAMES)


@pytest.mark.parametrize("bad_js", [0.0, -1.0, np.nan])
def test_invalid_raw_js_fails(tmp_path, bad_js):
    raw_list = tmp_path / "raw.txt"
    raw_log = tmp_path / "raw.log"
    write_raw_list(raw_list)
    write_raw_log(raw_log, raw_js=(25.0, bad_js))
    with pytest.raises(RuntimeError, match="J_raw values"):
        correction.build_matched_table(
            correction.load_clumpy_list(raw_list),
            correction.load_rendered_log(raw_log),
            np.asarray([0, 1]),
            np.asarray([100.0, 400.0]),
        )


def test_main_records_catalogue_provenance_and_required_csv_columns(
    tmp_path, monkeypatch
):
    catalog_root = tmp_path / "catalogues"
    clumpy_root = tmp_path / "clumpy"
    output_dir = tmp_path / "staged-correction"
    output_list = tmp_path / "staged-corrected-list.txt"
    paths = correction.get_paths(
        7, "resilient", 8, catalog_root=catalog_root, clumpy_root=clumpy_root
    )
    write_catalog(paths["catalog"])
    write_raw_list(paths["input_list"])
    write_raw_log(paths["rendered_log"])

    monkeypatch.setattr(correction, "make_plots", lambda *_: None)
    monkeypatch.setattr(
        correction,
        "parse_args",
        lambda: SimpleNamespace(
            repop_id=7,
            scenario="resilient",
            nside=8,
            catalog_root=catalog_root,
            clumpy_root=clumpy_root,
            raw_list=None,
            raw_rendered_log=None,
            output_list=output_list,
            output_dir=output_dir,
        ),
    )
    correction.main()

    csv_path = output_dir / "matched_R_diagnostics_resilient_repop_0007_nside8.csv"
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "catalog_path",
            "correction_target",
            "halo_name",
            "hdf5_row",
            "J_catalog",
            "J_raw",
            "R_raw",
            "correction_factor",
            "rhos_raw",
            "rhos_corrected",
        }
        assert required <= set(reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == 2
    assert {row["catalog_path"] for row in rows} == {str(paths["catalog"])}
    assert {row["correction_target"] for row in rows} == {
        "authoritative HDF5 catalogue Js"
    }
    text = output_list.read_text(encoding="utf-8")
    assert "Correction target: authoritative HDF5 catalogue Js" in text
    assert str(paths["catalog"]) in text


def test_validate_only_performs_matching_without_writing_outputs(
    tmp_path, monkeypatch
):
    catalog_root = tmp_path / "catalogues"
    clumpy_root = tmp_path / "clumpy"
    output_dir = tmp_path / "must-not-exist-diagnostics"
    output_list = tmp_path / "must-not-exist-corrected-list.txt"
    paths = correction.get_paths(
        7, "resilient", 8, catalog_root=catalog_root, clumpy_root=clumpy_root
    )
    write_catalog(paths["catalog"])
    write_raw_list(paths["input_list"])
    write_raw_log(paths["rendered_log"])

    monkeypatch.setattr(
        correction,
        "parse_args",
        lambda: SimpleNamespace(
            repop_id=7,
            scenario="resilient",
            nside=8,
            catalog_root=catalog_root,
            clumpy_root=clumpy_root,
            raw_list=None,
            raw_rendered_log=None,
            output_list=output_list,
            output_dir=output_dir,
            metadata_output_dir=None,
            validate_only=True,
        ),
    )

    correction.main()

    assert not output_dir.exists()
    assert not output_list.exists()

