from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "diagnostics"
    / "corrections"
    / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_clumpy_rendering_corrections as diagnostic  # noqa: E402


SCENARIO = "fragile"
REPOP_ID = 3
NSIDE = 1
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
CATALOG_NAMES = [
    "hydro_res_00001",
    "hydro_res_00002",
    "hydro_res_00003",
    "hydro_res_00004",
]
LIST_NAMES = CATALOG_NAMES[:2]


def write_list(path: Path, names: list[str], *, rhos_scale: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Name Type l b d z Rdelta rhos rs prof #1 #2 #3",
    ]
    for index, name in enumerate(names, start=1):
        lines.append(
            f"{name} DSPH {index}.0 0.0 10.0 -1 1.0 "
            f"{rhos_scale * index:.8e} 1.0 kZHAO 1 3 1"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_log(
    path: Path,
    names: list[str],
    rendered: list[float],
    *,
    indices: list[int] | None = None,
    malformed_line: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if indices is None:
        indices = list(range(len(names)))
    lines = ["# HALO_INDEX\tNAME\tTYPE\tJ_RENDERED\tNPIX_TOUCHED"]
    for index, name, value in zip(indices, names, rendered):
        lines.append(f"{index}\t{name}\tDSPH\t{value:.16g}\t4")
    if malformed_line is not None:
        lines.append(malformed_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_catalog(
    path: Path,
    names: list[str],
    js: list[float],
    theta: list[float],
    *,
    n_saved: int | None = None,
    halo_name_count: int | None = None,
    columns: list[str] | None = None,
    units: list[str] | None = None,
    chunk_rows: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = COLUMNS.copy() if columns is None else columns
    units = UNITS.copy() if units is None else units
    data = np.ones((len(names), len(columns)), dtype=np.float64)
    if "Js" in columns:
        data[:, columns.index("Js")] = js
    if "theta_s" in columns:
        data[:, columns.index("theta_s")] = theta
    with h5py.File(path, "w") as handle:
        group = handle.create_group("iteration_0")
        dataset = group.create_dataset(
            "data",
            data=data,
            chunks=(min(chunk_rows, len(names)), len(columns)),
        )
        dataset.attrs["column_names"] = np.asarray(
            columns, dtype=h5py.string_dtype("utf-8")
        )
        dataset.attrs["units"] = np.asarray(
            units, dtype=h5py.string_dtype("utf-8")
        )
        count = len(names) if halo_name_count is None else halo_name_count
        stored_names = names[:count]
        if count > len(names):
            stored_names += [f"extra_{index}" for index in range(count - len(names))]
        group.create_dataset(
            "halo_name",
            data=np.asarray(stored_names, dtype=h5py.string_dtype("utf-8")),
            chunks=(min(chunk_rows, count),),
        )
        group.attrs["n_saved"] = len(names) if n_saved is None else n_saved


def write_sparse_catalog(
    path: Path,
    n_rows: int,
    entries: dict[int, tuple[str, float, float]],
    *,
    chunk_rows: int = 128,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("iteration_0")
        data = group.create_dataset(
            "data",
            shape=(n_rows, len(COLUMNS)),
            dtype=np.float64,
            chunks=(chunk_rows, len(COLUMNS)),
            fillvalue=1.0,
        )
        data.attrs["column_names"] = np.asarray(
            COLUMNS, dtype=h5py.string_dtype("utf-8")
        )
        data.attrs["units"] = np.asarray(
            UNITS, dtype=h5py.string_dtype("utf-8")
        )
        names = group.create_dataset(
            "halo_name",
            shape=(n_rows,),
            dtype=h5py.string_dtype("utf-8"),
            chunks=(chunk_rows,),
        )
        for row_index, (name, js, theta) in entries.items():
            names[row_index] = name
            data[row_index, COLUMNS.index("Js")] = js
            data[row_index, COLUMNS.index("theta_s")] = theta
        group.attrs["n_saved"] = n_rows


def make_map_hdu(
    name: str,
    nside: int,
    *,
    per_sr: bool,
    column_order: list[str] | None = None,
    extra_column: bool = False,
    n_rows: int | None = None,
) -> fits.BinTableHDU:
    npix = 12 * nside * nside if n_rows is None else n_rows
    pixel = np.arange(npix, dtype=np.int32)
    suffix = "_per_sr" if per_sr else ""
    columns = {"PIXEL": fits.Column(name="PIXEL", format="1J", array=pixel)}
    for base in ("Jtot", "Jsmooth", "Jextended", "Jpointlike"):
        column_name = f"{base}{suffix}"
        columns[column_name] = fits.Column(
            name=column_name,
            format="1E",
            array=np.ones(npix, dtype=np.float32),
        )
    if extra_column:
        columns["EXTRA"] = fits.Column(
            name="EXTRA",
            format="1E",
            array=np.zeros(npix, dtype=np.float32),
        )
    ordered_names = list(columns) if column_order is None else column_order
    hdu = fits.BinTableHDU.from_columns(
        [columns[column_name] for column_name in ordered_names],
        name=name,
    )
    hdu.header["NSIDE"] = nside
    return hdu


def write_total_fits(path: Path, nside: int = NSIDE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            make_map_hdu("JFACTOR", nside, per_sr=False),
            make_map_hdu("JFACTOR_PER_SR", nside, per_sr=True),
        ]
    ).writeto(path)


def make_valid_case(
    tmp_path: Path,
    *,
    catalog_names: list[str] | None = None,
    list_names: list[str] | None = None,
    js: list[float] | None = None,
    theta: list[float] | None = None,
    raw_rendered: list[float] | None = None,
    corr_rendered: list[float] | None = None,
) -> tuple[Path, Path, diagnostic.CasePaths]:
    catalog_root = tmp_path / "catalogues"
    clumpy_root = tmp_path / "clumpy"
    catalog_names = catalog_names or CATALOG_NAMES.copy()
    list_names = list_names or LIST_NAMES.copy()
    js = js or [10.0, 20.0, 5.0, 7.0]
    theta = theta or [0.20, 0.40, 0.01, 0.02]
    raw_rendered = raw_rendered or [8.0, 24.0]
    corr_rendered = corr_rendered or [10.0, 20.0]
    paths = diagnostic.build_case_paths(
        catalog_root, clumpy_root, SCENARIO, REPOP_ID, NSIDE
    )
    write_catalog(paths.catalog, catalog_names, js, theta)
    write_list(paths.raw_list, list_names)
    write_list(paths.corrected_list, list_names, rhos_scale=1.1)
    write_log(paths.raw_log, list_names, raw_rendered)
    write_log(paths.corrected_log, list_names, corr_rendered)
    write_total_fits(paths.total_fits)
    return catalog_root, clumpy_root, paths


def validate(
    catalog_root: Path,
    clumpy_root: Path,
) -> tuple[diagnostic.ValidationRecord, diagnostic.CaseData | None]:
    return diagnostic.validate_case(
        catalog_root,
        clumpy_root,
        SCENARIO,
        REPOP_ID,
        NSIDE,
    )


def synthetic_case(
    scenario: str,
    repop_id: int,
    r_values: np.ndarray,
) -> diagnostic.CaseData:
    r_values = np.asarray(r_values, dtype=np.float64)
    count = len(r_values)
    ones = np.ones(count, dtype=np.float64)
    return diagnostic.CaseData(
        scenario=scenario,
        repop_id=repop_id,
        names=tuple(f"h{index}" for index in range(count)),
        j_catalog=ones,
        theta_s_deg=np.arange(1, count + 1, dtype=float),
        j_rendered_raw=r_values.copy(),
        j_rendered_corr=r_values.copy(),
        x=np.arange(1, count + 1, dtype=float),
        r_raw=r_values.copy(),
        r_corr=r_values.copy(),
        delta_raw=100.0 * np.abs(r_values - 1.0),
        delta_corr=100.0 * np.abs(r_values - 1.0),
    )


def test_valid_matching_indexed_lookup_and_ratios(tmp_path):
    catalog_root, clumpy_root, _ = make_valid_case(tmp_path)
    record, case = validate(catalog_root, clumpy_root)

    assert record.usable
    assert record.n_matched == 2
    assert case is not None
    assert case.names == tuple(LIST_NAMES)
    np.testing.assert_allclose(case.j_catalog, [10.0, 20.0])
    np.testing.assert_allclose(case.r_raw, [0.8, 1.2])
    np.testing.assert_allclose(case.r_corr, [1.0, 1.0])
    assert np.max(case.r_raw) == pytest.approx(1.2)
    np.testing.assert_allclose(case.delta_corr, [0.0, 0.0])
    expected_theta_p = np.degrees(np.sqrt(4.0 * np.pi / 12.0))
    np.testing.assert_allclose(case.x, np.asarray([0.2, 0.4]) / expected_theta_p)


def test_zero_based_halo_index_is_required(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_log(paths.raw_log, LIST_NAMES, [8.0, 24.0], indices=[1, 2])
    record, case = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert case is None
    assert "zero-based" in record.exclusion_reason


@pytest.mark.parametrize("target", ["raw_list", "raw_log", "corrected_log"])
def test_duplicate_names_are_rejected(tmp_path, target):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    if target == "raw_list":
        write_list(paths.raw_list, [LIST_NAMES[0], LIST_NAMES[0]])
    elif target == "raw_log":
        write_log(paths.raw_log, [LIST_NAMES[0], LIST_NAMES[0]], [8.0, 24.0])
    else:
        write_log(
            paths.corrected_log,
            [LIST_NAMES[0], LIST_NAMES[0]],
            [10.0, 20.0],
        )
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert "duplicate NAME" in record.exclusion_reason


def test_stored_hdf5_name_mismatch_is_rejected(tmp_path):
    catalog_root, clumpy_root, _ = make_valid_case(
        tmp_path,
        catalog_names=[
            "hydro_res_00001",
            "not_hydro_res_00002",
            "hydro_res_00003",
            "hydro_res_00004",
        ],
    )
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert "HDF5 NAME mismatch at inferred row 1" in record.exclusion_reason


def test_reordered_corrected_list_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_list(paths.corrected_list, list(reversed(LIST_NAMES)))
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert "differs from raw list" in record.exclusion_reason


@pytest.mark.parametrize("stage", ["raw", "corrected"])
def test_partial_logs_are_rejected(tmp_path, stage):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    target = paths.raw_log if stage == "raw" else paths.corrected_log
    write_log(target, [LIST_NAMES[0]], [10.0])
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert "row-count mismatch" in record.exclusion_reason


def test_malformed_rendered_log_row_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_log(
        paths.raw_log,
        LIST_NAMES,
        [8.0, 24.0],
        malformed_line="2 only three fields",
    )
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert "Malformed rendered-log row" in record.exclusion_reason


def test_missing_completion_fits_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    paths.total_fits.unlink()
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert not record.total_fits_valid
    assert "FITS is missing" in record.exclusion_reason


def test_invalid_truncated_completion_fits_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    complete = paths.total_fits.read_bytes()
    paths.total_fits.write_bytes(complete[: len(complete) // 2])
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert not record.total_fits_valid
    assert "JFACTOR_PER_SR" in record.exclusion_reason


def test_completion_fits_accepts_reordered_extension_positions(tmp_path):
    path = tmp_path / "reordered_extensions.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            make_map_hdu("JFACTOR_PER_SR", NSIDE, per_sr=True),
            fits.ImageHDU(name="HARMLESS"),
            make_map_hdu("JFACTOR", NSIDE, per_sr=False),
        ]
    ).writeto(path)

    diagnostic.validate_total_fits(path, NSIDE)


def test_completion_fits_accepts_reordered_required_columns(tmp_path):
    path = tmp_path / "reordered_columns.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            make_map_hdu(
                "JFACTOR",
                NSIDE,
                per_sr=False,
                column_order=[
                    "Jextended",
                    "PIXEL",
                    "Jpointlike",
                    "Jtot",
                    "Jsmooth",
                ],
            ),
            make_map_hdu(
                "JFACTOR_PER_SR",
                NSIDE,
                per_sr=True,
                column_order=[
                    "Jextended_per_sr",
                    "PIXEL",
                    "Jpointlike_per_sr",
                    "Jtot_per_sr",
                    "Jsmooth_per_sr",
                ],
            ),
        ]
    ).writeto(path)

    diagnostic.validate_total_fits(path, NSIDE)


def test_completion_fits_accepts_additional_columns(tmp_path):
    path = tmp_path / "additional_columns.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            make_map_hdu(
                "JFACTOR", NSIDE, per_sr=False, extra_column=True
            ),
            make_map_hdu(
                "JFACTOR_PER_SR", NSIDE, per_sr=True, extra_column=True
            ),
        ]
    ).writeto(path)

    diagnostic.validate_total_fits(path, NSIDE)


def test_completion_fits_accepts_additional_hdus(tmp_path):
    path = tmp_path / "additional_hdus.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(name="BEFORE"),
            make_map_hdu("JFACTOR", NSIDE, per_sr=False),
            fits.ImageHDU(name="BETWEEN"),
            make_map_hdu("JFACTOR_PER_SR", NSIDE, per_sr=True),
            fits.ImageHDU(name="AFTER"),
        ]
    ).writeto(path)

    diagnostic.validate_total_fits(path, NSIDE)


def test_completion_fits_missing_required_extension_is_rejected(tmp_path):
    path = tmp_path / "missing_extension.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            make_map_hdu("JFACTOR", NSIDE, per_sr=False),
        ]
    ).writeto(path)

    with pytest.raises(
        diagnostic.ProductValidationError,
        match="exactly one JFACTOR_PER_SR extension; found 0",
    ):
        diagnostic.validate_total_fits(path, NSIDE)


def test_completion_fits_duplicate_required_extension_is_rejected(tmp_path):
    path = tmp_path / "duplicate_extension.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            make_map_hdu("JFACTOR", NSIDE, per_sr=False),
            make_map_hdu("JFACTOR", NSIDE, per_sr=False),
            make_map_hdu("JFACTOR_PER_SR", NSIDE, per_sr=True),
        ]
    ).writeto(path)

    with pytest.raises(
        diagnostic.ProductValidationError,
        match="exactly one JFACTOR extension; found 2",
    ):
        diagnostic.validate_total_fits(path, NSIDE)


def test_completion_fits_missing_required_column_is_rejected(tmp_path):
    path = tmp_path / "missing_column.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            make_map_hdu(
                "JFACTOR",
                NSIDE,
                per_sr=False,
                column_order=["PIXEL", "Jtot", "Jsmooth", "Jextended"],
            ),
            make_map_hdu("JFACTOR_PER_SR", NSIDE, per_sr=True),
        ]
    ).writeto(path)

    with pytest.raises(
        diagnostic.ProductValidationError,
        match=r"missing required column\(s\): Jpointlike",
    ):
        diagnostic.validate_total_fits(path, NSIDE)


def test_completion_fits_wrong_number_of_rows_is_rejected(tmp_path):
    path = tmp_path / "wrong_rows.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            make_map_hdu(
                "JFACTOR", NSIDE, per_sr=False, n_rows=11
            ),
            make_map_hdu("JFACTOR_PER_SR", NSIDE, per_sr=True),
        ]
    ).writeto(path)

    with pytest.raises(
        diagnostic.ProductValidationError,
        match="has 11 rows, expected 12",
    ):
        diagnostic.validate_total_fits(path, NSIDE)


def test_wrong_completion_fits_nside_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    paths.total_fits.unlink()
    write_total_fits(paths.total_fits, nside=2)
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert "NSIDE=2" in record.exclusion_reason


def test_hdf5_n_saved_mismatch_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_catalog(
        paths.catalog,
        CATALOG_NAMES,
        [10.0, 20.0, 5.0, 7.0],
        [0.2, 0.4, 0.01, 0.02],
        n_saved=99,
    )
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert "n_saved mismatch" in record.exclusion_reason


def test_hdf5_halo_name_length_mismatch_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    write_catalog(
        paths.catalog,
        CATALOG_NAMES,
        [10.0, 20.0, 5.0, 7.0],
        [0.2, 0.4, 0.01, 0.02],
        halo_name_count=3,
    )
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert "halo_name length mismatch" in record.exclusion_reason


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_js", "Missing HDF5 column"),
        ("wrong_js_unit", "Wrong unit for Js"),
        ("wrong_theta_unit", "Wrong unit for theta_s"),
    ],
)
def test_hdf5_missing_or_wrong_units_are_rejected(tmp_path, mutation, message):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    columns = COLUMNS.copy()
    units = UNITS.copy()
    if mutation == "missing_js":
        index = columns.index("Js")
        columns[index] = "not_Js"
    elif mutation == "wrong_js_unit":
        units[columns.index("Js")] = "GeV"
    else:
        units[columns.index("theta_s")] = "rad"
    write_catalog(
        paths.catalog,
        CATALOG_NAMES,
        [10.0, 20.0, 5.0, 7.0],
        [0.2, 0.4, 0.01, 0.02],
        columns=columns,
        units=units,
    )
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert message in record.exclusion_reason


def test_hdf5_missing_units_metadata_is_rejected(tmp_path):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    with h5py.File(paths.catalog, "r+") as handle:
        del handle["iteration_0/data"].attrs["units"]
    record, _ = validate(catalog_root, clumpy_root)
    assert not record.usable
    assert "Missing column_names or units metadata" in record.exclusion_reason


def test_indexed_lookup_maps_first_two_names_to_rows_zero_and_one(tmp_path):
    path = tmp_path / "catalog.h5"
    write_catalog(
        path,
        CATALOG_NAMES,
        [11.0, 22.0, 33.0, 44.0],
        [0.11, 0.22, 0.33, 0.44],
    )

    js, theta = diagnostic.load_catalog_values_indexed(path, LIST_NAMES)

    np.testing.assert_allclose(js, [11.0, 22.0])
    np.testing.assert_allclose(theta, [0.11, 0.22])


def test_indexed_lookup_accepts_suffix_above_five_digits(tmp_path):
    path = tmp_path / "large_sparse_catalog.h5"
    write_sparse_catalog(
        path,
        100_000,
        {99_999: ("hydro_res_100000", 42.0, 0.75)},
    )

    js, theta = diagnostic.load_catalog_values_indexed(
        path, ["hydro_res_100000"]
    )

    np.testing.assert_allclose(js, [42.0])
    np.testing.assert_allclose(theta, [0.75])


@pytest.mark.parametrize(
    "name",
    ["halo_a", "hydro_frag_00001", "hydro_res_1x", "hydro_res_-1"],
)
def test_indexed_lookup_rejects_malformed_names(tmp_path, name):
    path = tmp_path / "catalog.h5"
    write_catalog(path, CATALOG_NAMES, [1.0] * 4, [0.1] * 4)

    with pytest.raises(
        diagnostic.ProductValidationError,
        match="Non-canonical halo NAME",
    ):
        diagnostic.load_catalog_values_indexed(path, [name])


def test_indexed_lookup_rejects_zero_suffix(tmp_path):
    path = tmp_path / "catalog.h5"
    write_catalog(path, CATALOG_NAMES, [1.0] * 4, [0.1] * 4)

    with pytest.raises(
        diagnostic.ProductValidationError,
        match="Non-canonical halo NAME",
    ):
        diagnostic.load_catalog_values_indexed(path, ["hydro_res_00000"])


def test_indexed_lookup_rejects_row_outside_catalog(tmp_path):
    path = tmp_path / "catalog.h5"
    write_catalog(path, CATALOG_NAMES, [1.0] * 4, [0.1] * 4)

    with pytest.raises(
        diagnostic.ProductValidationError,
        match="outside valid range 0..3",
    ):
        diagnostic.load_catalog_values_indexed(path, ["hydro_res_00005"])


def test_indexed_lookup_rejects_duplicate_inferred_rows(tmp_path):
    path = tmp_path / "catalog.h5"
    write_catalog(path, CATALOG_NAMES, [1.0] * 4, [0.1] * 4)

    with pytest.raises(
        diagnostic.ProductValidationError,
        match="Duplicate inferred HDF5 row 0",
    ):
        diagnostic.load_catalog_values_indexed(
            path, ["hydro_res_00001", "hydro_res_1"]
        )


def test_indexed_lookup_restores_unsorted_list_order_for_separated_rows(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "catalog.h5"
    names = [f"hydro_res_{index:05d}" for index in range(1, 9)]
    write_catalog(
        path,
        names,
        [float(index) for index in range(1, 9)],
        [float(index) / 10.0 for index in range(1, 9)],
        chunk_rows=2,
    )
    decoded_halo_names = 0
    original_decode = diagnostic._decode_strings

    def recording_decode(values):
        nonlocal decoded_halo_names
        decoded = original_decode(values)
        if decoded and all(
            value.startswith("hydro_res_") for value in decoded
        ):
            decoded_halo_names += len(decoded)
        return decoded

    monkeypatch.setattr(diagnostic, "_decode_strings", recording_decode)
    requested = ["hydro_res_00008", "hydro_res_00001"]

    js, theta = diagnostic.load_catalog_values_indexed(path, requested)

    np.testing.assert_allclose(js, [8.0, 1.0])
    np.testing.assert_allclose(theta, [0.8, 0.1])
    assert decoded_halo_names == 2
    assert decoded_halo_names < len(names)


def test_per_repopulation_summary_exact_equality_and_strict_fractions():
    case = synthetic_case("fragile", 1, np.asarray([1.0, 1.0, 1.2]))
    summary = diagnostic.per_repopulation_summary(case)
    assert summary["delta_raw_p68"] >= 0.0
    assert summary["frac_delta_raw_lt_1pct"] == pytest.approx(2.0 / 3.0)
    exact = synthetic_case("fragile", 2, np.ones(4))
    exact_summary = diagnostic.per_repopulation_summary(exact)
    assert exact_summary["r_corr_median"] == 1.0
    assert exact_summary["delta_corr_p99"] == 0.0
    assert exact_summary["frac_delta_corr_lt_1pct"] == 1.0


def test_equal_repopulation_weighting_with_unequal_halo_counts():
    small = synthetic_case("fragile", 1, np.asarray([0.0]))
    large = synthetic_case("fragile", 2, np.full(100, 10.0))
    summaries = [
        diagnostic.per_repopulation_summary(small),
        diagnostic.per_repopulation_summary(large),
    ]
    rows = diagnostic.ensemble_summary_rows(summaries)
    median_row = next(row for row in rows if row["statistic"] == "r_raw_median")
    assert median_row["ensemble_median"] == pytest.approx(5.0)
    assert median_row["total_matched_halos"] == 101
    pooled_median = np.median(np.concatenate([small.r_raw, large.r_raw]))
    assert pooled_median == 10.0
    assert median_row["ensemble_median"] != pooled_median


def test_per_repopulation_binned_medians_and_ensemble_bands():
    edges = np.asarray([1.0, 10.0, 100.0])
    first = diagnostic.binned_medians(
        np.asarray([1.0, 2.0, 20.0]),
        np.asarray([1.0, 3.0, 8.0]),
        edges,
    )
    second = diagnostic.binned_medians(
        np.asarray([5.0, 50.0]),
        np.asarray([5.0, 10.0]),
        edges,
    )
    np.testing.assert_allclose(first, [2.0, 8.0])
    np.testing.assert_allclose(second, [5.0, 10.0])
    lower, median, upper, count = diagnostic.pointwise_percentiles(
        np.asarray([first, second])
    )
    np.testing.assert_allclose(median, [3.5, 9.0])
    np.testing.assert_allclose(
        lower,
        np.percentile([[2.0, 8.0], [5.0, 10.0]], 16, axis=0),
    )
    np.testing.assert_allclose(upper, np.percentile([[2.0, 8.0], [5.0, 10.0]], 84, axis=0))
    np.testing.assert_array_equal(count, [2, 2])


def test_empirical_cdf_on_common_grid_includes_zero_residuals():
    grid = np.asarray([0.0, 0.5, 1.0, 2.0])
    first = diagnostic.empirical_cdf(np.asarray([0.0, 1.0, 2.0]), grid)
    second = diagnostic.empirical_cdf(np.asarray([0.0, 0.0, 2.0]), grid)
    np.testing.assert_allclose(first, [1 / 3, 1 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(second, [2 / 3, 2 / 3, 2 / 3, 1.0])
    lower, median, upper, count = diagnostic.pointwise_percentiles(
        np.asarray([first, second])
    )
    np.testing.assert_allclose(median, [0.5, 0.5, 2 / 3, 1.0])
    assert np.all(lower <= median)
    assert np.all(median <= upper)
    np.testing.assert_array_equal(count, [2, 2, 2, 2])


def test_validate_only_uses_indexed_lookup_and_writes_nothing(
    tmp_path,
    capsys,
    monkeypatch,
):
    catalog_root, clumpy_root, paths = make_valid_case(tmp_path)
    paths.total_fits.unlink()
    output_dir = tmp_path / "must_not_exist"
    decoded_halo_names = 0
    original_decode = diagnostic._decode_strings

    def recording_decode(values):
        nonlocal decoded_halo_names
        decoded = original_decode(values)
        if decoded and all(
            value.startswith("hydro_res_") for value in decoded
        ):
            decoded_halo_names += len(decoded)
        return decoded

    monkeypatch.setattr(diagnostic, "_decode_strings", recording_decode)
    status = diagnostic.main(
        [
            "--catalog-root",
            str(catalog_root),
            "--clumpy-root",
            str(clumpy_root),
            "--scenario",
            SCENARIO,
            "--repop-ids",
            str(REPOP_ID),
            "--nside",
            str(NSIDE),
            "--output-dir",
            str(output_dir),
            "--validate-only",
        ]
    )
    output = capsys.readouterr().out
    assert status == 1
    assert "scenario=fragile" in output
    assert "total_fits_valid=no" in output
    assert "hdf5_matching=yes" in output
    assert "n_matched=2" in output
    assert "usable=no" in output
    assert "exclusion_reason=" in output
    assert decoded_halo_names == 2
    assert decoded_halo_names < len(CATALOG_NAMES)
    assert not (output_dir / "matched_halo_diagnostics.csv").exists()
    assert not output_dir.exists()


def test_half_open_range_and_repop_id_exclusivity():
    args = diagnostic.parse_args(
        [
            "--catalog-root",
            "/catalog",
            "--clumpy-root",
            "/clumpy",
            "--scenario",
            "fragile",
            "--start-id",
            "0",
            "--stop-id",
            "3",
        ]
    )
    assert diagnostic.parse_repop_ids(args) == [0, 1, 2]

    explicit = diagnostic.parse_args(
        [
            "--catalog-root",
            "/catalog",
            "--clumpy-root",
            "/clumpy",
            "--scenario",
            "fragile",
            "--repop-ids",
            "1,4",
            "7",
        ]
    )
    assert diagnostic.parse_repop_ids(explicit) == [1, 4, 7]


def test_csv_outputs_contain_requested_fields(tmp_path):
    case = synthetic_case("resilient", 7, np.asarray([0.9, 1.0, 1.1]))
    per_repop = [diagnostic.per_repopulation_summary(case)]
    ensemble = diagnostic.ensemble_summary_rows(per_repop)
    per_path = tmp_path / "per_repopulation_summary.csv"
    ensemble_path = tmp_path / "ensemble_summary.csv"
    diagnostic.write_csv(per_path, per_repop)
    diagnostic.write_csv(ensemble_path, ensemble)
    with per_path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["scenario"] == "resilient"
    assert "delta_raw_p99" in row
    assert "frac_delta_corr_lt_10pct" in row
    with ensemble_path.open(newline="") as stream:
        ensemble_row = next(csv.DictReader(stream))
    assert set(
        ["ensemble_median", "ensemble_p16", "ensemble_p84", "total_matched_halos"]
    ).issubset(ensemble_row)


def test_full_analysis_writes_required_outputs(tmp_path, monkeypatch):
    catalog_root, clumpy_root, _ = make_valid_case(tmp_path)
    output_dir = tmp_path / "analysis"

    def fake_figure(trend_rows, cdf_rows, target, nside):
        assert trend_rows
        assert cdf_rows
        png = target / f"clumpy_rendering_corrections_nside{nside}.png"
        pdf = target / f"clumpy_rendering_corrections_nside{nside}.pdf"
        png.write_bytes(b"png")
        pdf.write_bytes(b"pdf")
        return png, pdf

    monkeypatch.setattr(diagnostic, "make_figure", fake_figure)
    status = diagnostic.main(
        [
            "--catalog-root",
            str(catalog_root),
            "--clumpy-root",
            str(clumpy_root),
            "--scenario",
            SCENARIO,
            "--repop-ids",
            str(REPOP_ID),
            "--nside",
            str(NSIDE),
            "--x-bins",
            "3",
            "--cdf-points",
            "8",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0
    for name in (
        "matched_halo_diagnostics.csv",
        "per_repopulation_summary.csv",
        "ensemble_summary.csv",
        "binned_trend_summary.csv",
        "cdf_summary.csv",
        "clumpy_rendering_corrections_nside1.png",
        "clumpy_rendering_corrections_nside1.pdf",
    ):
        assert (output_dir / name).is_file()


def test_matched_halo_csv_schema_values_and_order(tmp_path, monkeypatch):
    catalog_root, clumpy_root, _ = make_valid_case(tmp_path)
    output_dir = tmp_path / "analysis"

    def fake_figure(trend_rows, cdf_rows, target, nside):
        png = target / f"clumpy_rendering_corrections_nside{nside}.png"
        pdf = target / f"clumpy_rendering_corrections_nside{nside}.pdf"
        png.write_bytes(b"png")
        pdf.write_bytes(b"pdf")
        return png, pdf

    monkeypatch.setattr(diagnostic, "make_figure", fake_figure)
    status = diagnostic.main(
        [
            "--catalog-root",
            str(catalog_root),
            "--clumpy-root",
            str(clumpy_root),
            "--scenario",
            SCENARIO,
            "--repop-ids",
            str(REPOP_ID),
            "--nside",
            str(NSIDE),
            "--x-bins",
            "3",
            "--cdf-points",
            "8",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert status == 0

    path = output_dir / "matched_halo_diagnostics.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "scenario",
            "repop_id",
            "halo_name",
            "theta_s_deg",
            "theta_p_deg",
            "theta_s_over_theta_p",
            "J_catalog",
            "J_raw",
            "R_raw",
            "delta_raw_percent",
            "J_corr",
            "R_corr",
            "delta_corr_percent",
        ]
        rows = list(reader)

    assert [row["halo_name"] for row in rows] == LIST_NAMES
    assert [row["scenario"] for row in rows] == [SCENARIO, SCENARIO]
    assert [int(row["repop_id"]) for row in rows] == [REPOP_ID, REPOP_ID]

    theta_p = np.degrees(np.sqrt(4.0 * np.pi / (12.0 * NSIDE**2)))
    expected = [
        (0.2, 10.0, 8.0, 0.8, 10.0, 1.0),
        (0.4, 20.0, 24.0, 1.2, 20.0, 1.0),
    ]
    for row, (
        theta_s,
        j_catalog,
        j_raw,
        r_raw,
        j_corr,
        r_corr,
    ) in zip(rows, expected):
        assert float(row["theta_s_deg"]) == pytest.approx(theta_s)
        assert float(row["theta_p_deg"]) == pytest.approx(theta_p)
        assert float(row["theta_s_over_theta_p"]) == pytest.approx(
            theta_s / theta_p
        )
        assert float(row["J_catalog"]) == pytest.approx(j_catalog)
        assert float(row["J_raw"]) == pytest.approx(j_raw)
        assert float(row["R_raw"]) == pytest.approx(r_raw)
        assert float(row["delta_raw_percent"]) == pytest.approx(
            100.0 * abs(r_raw - 1.0)
        )
        assert float(row["J_corr"]) == pytest.approx(j_corr)
        assert float(row["R_corr"]) == pytest.approx(r_corr)
        assert float(row["delta_corr_percent"]) == pytest.approx(
            100.0 * abs(r_corr - 1.0)
        )


def test_four_panel_figure_writes_without_external_latex(tmp_path, monkeypatch):
    import matplotlib
    from matplotlib import texmanager

    def fail_if_external_tex_is_used(*args, **kwargs):
        raise AssertionError("External LaTeX rendering was attempted")

    monkeypatch.setenv("PATH", str(tmp_path / "no-executables"))
    monkeypatch.setattr(
        texmanager.TexManager,
        "make_dvi",
        fail_if_external_tex_is_used,
    )
    cases = [
        synthetic_case("fragile", 1, np.asarray([0.9, 1.0, 1.1])),
        synthetic_case("resilient", 2, np.asarray([0.8, 1.0, 1.2])),
    ]
    edges = diagnostic.common_log_edges(cases, 3)
    trend_rows, _ = diagnostic.build_trend_summaries(cases, edges)
    grid = diagnostic.common_cdf_grid(cases, 12)
    cdf_rows, _ = diagnostic.build_cdf_summaries(cases, grid)
    png, pdf = diagnostic.make_figure(trend_rows, cdf_rows, tmp_path, NSIDE)
    assert matplotlib.rcParams["text.usetex"] is False
    assert matplotlib.rcParams["mathtext.fontset"] == "stix"
    assert png.is_file() and png.stat().st_size > 0
    assert pdf.is_file() and pdf.stat().st_size > 0
