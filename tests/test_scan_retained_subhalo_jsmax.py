import csv
import gzip
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "diagnostics"
    / "catalogue"
    / "scripts"
    / "scan_retained_subhalo_jsmax.py"
)
SPEC = importlib.util.spec_from_file_location("scan_retained_subhalo_jsmax", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


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
    "GeV2 / cm5",
    "kpc",
    "km / s",
    "deg",
    "",
    "kpc",
    "solMass / kpc3",
    "kpc",
    "kpc",
    "kpc",
]
NAMES = [f"hydro_res_{index:05d}" for index in range(1, 7)]


def dark_rows():
    return [
        [900.0, 10.0, 9.0, 0.5, 1.0, 1.0, 2.0, 10.0, 0.0, 0.0],
        [200.0, 11.0, 8.0, 0.4, 1.0, 1.0, 2.0, 0.0, 11.0, 0.0],
        [500.0, 12.0, 8.0, 2.0, 1.0, 1.0, 2.0, -12.0, 0.0, 0.0],
        [1000.0, 13.0, 1.0, 0.3, 1.0, 1.0, np.nan, 0.0, -13.0, 0.0],
        [300.0, 14.0, 4.0, 3.0, 1.0, 1.0, 2.0, 0.0, 0.0, 14.0],
        [400.0, 15.0, 4.0, 0.2, 1.0, 1.0, 2.0, 15.0, 0.0, 0.0],
    ]


def write_catalog(path: Path, rows=None) -> None:
    data = np.asarray(
        rows
        if rows is not None
        else [
            [200.0, 10.0, 1.0, 0.5, 1.0, 1.0, 2.0, 10.0, 0.0, 0.0],
            [50.0, 11.0, 1.0, 0.4, 1.0, 1.0, 2.0, 0.0, 11.0, 0.0],
            [500.0, 12.0, 1.0, 2.0, 1.0, 1.0, 2.0, -12.0, 0.0, 0.0],
            [1000.0, 13.0, 1.0, 0.3, 1.0, 1.0, np.nan, 0.0, -13.0, 0.0],
            [300.0, 14.0, 1.0, 3.0, 1.0, 1.0, 2.0, 0.0, 0.0, 14.0],
            [400.0, 15.0, 1.0, 0.2, 1.0, 1.0, 2.0, 15.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("iteration_0")
        dataset = group.create_dataset("data", data=data)
        dataset.attrs["column_names"] = np.asarray(
            COLUMNS, dtype=h5py.string_dtype("utf-8")
        )
        dataset.attrs["units"] = np.asarray(
            UNITS, dtype=h5py.string_dtype("utf-8")
        )
        group.create_dataset(
            "halo_name",
            data=np.asarray(NAMES, dtype=h5py.string_dtype("utf-8")),
        )
        group.attrs["n_saved"] = len(data)


def write_pointlike_fits(
    path: Path,
    npoint: int = 2,
    physical_selection: str = "legacy",
    max_vmax_kms: float = 8.0,
) -> None:
    path.parent.mkdir(parents=True)
    hdus = [fits.PrimaryHDU()]
    for name in ("JFACTOR", "JFACTOR_PER_SR"):
        hdu = fits.BinTableHDU.from_columns([], name=name)
        hdu.header["NSIDE"] = 8
        hdu.header["SCENARIO"] = "fragile"
        hdu.header["REPOPID"] = 0
        hdu.header["CUTS"] = True
        hdu.header["THETACUT"] = 1.0
        hdu.header["FPL"] = 0.1
        hdu.header["JREF"] = 1000.0
        hdu.header["JCPL"] = 100.0
        hdu.header["FEXT"] = 0.1
        hdu.header["JCEXT"] = 100.0
        hdu.header["NPOINT"] = npoint
        if physical_selection == "full":
            hdu.header["VMAXCUT"] = False
        elif physical_selection == "selected":
            hdu.header["VMAXCUT"] = True
            hdu.header["VMAXMAX"] = max_vmax_kms
        elif physical_selection != "legacy":
            raise ValueError(f"unknown physical selection {physical_selection!r}")
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path)


def list_line(name: str) -> str:
    return f"{name} DSPH 0 0 10 -1 1 1 1 kZHAO 1 3 1\n"


def write_extended_products(
    list_path: Path,
    log_path: Path,
    reverse_log=False,
    scenario="fragile",
    repop_id=0,
    nside=8,
    iteration=0,
    source_hdf5="/production/repop_0000/fullrepop_hydro_fragile.h5",
    physical_selection="legacy",
    max_vmax_kms=8.0,
    extended=None,
) -> None:
    extended = list(extended if extended is not None else [NAMES[2], NAMES[4]])
    list_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# corrected list\n"
        f"# Source HDF5: {source_hdf5}\n"
        f"# Scenario: {scenario}\n"
        f"# REPOP_ID: {repop_id}\n"
        f"# HDF5 internal iteration: {iteration}\n"
        f"# HEALPix NSIDE: {nside}\n"
    )
    if physical_selection == "full":
        header += "# Physical Vmax selection: disabled (full population)\n"
    elif physical_selection == "selected":
        header += (
            f"# Physical Vmax selection: Vmax <= {max_vmax_kms} km/s\n"
        )
    elif physical_selection != "legacy":
        raise ValueError(f"unknown physical selection {physical_selection!r}")
    list_path.write_text(
        header + "".join(list_line(name) for name in extended),
        encoding="utf-8",
    )
    if reverse_log:
        extended.reverse()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# HALO_INDEX NAME TYPE J_RENDERED NPIX_TOUCHED\n"]
    lines.extend(
        f"{index} {name} DSPH {100 + index:.1f} {index + 1}\n"
        for index, name in enumerate(extended)
    )
    log_path.write_text("".join(lines), encoding="utf-8")


def make_case(
    tmp_path: Path,
    npoint: int = 2,
    reverse_log=False,
    physical_selection="legacy",
    max_vmax_kms=8.0,
    rows=None,
    extended=None,
):
    output_dir = tmp_path / "results"
    if physical_selection == "selected":
        value = str(max_vmax_kms).rstrip("0").rstrip(".").replace(".", "p")
        output_dir = output_dir / f"dark_vmax_le_{value}kms"
    paths = diagnostic.case_paths(
        tmp_path / "catalogues",
        tmp_path / "clumpy",
        output_dir,
        "fragile",
        0,
        8,
    )
    write_catalog(paths.catalog, rows=rows)
    write_pointlike_fits(
        paths.pointlike_fits,
        npoint=npoint,
        physical_selection=physical_selection,
        max_vmax_kms=max_vmax_kms,
    )
    write_extended_products(
        paths.corrected_list,
        paths.corrected_rendered_log,
        reverse_log=reverse_log,
        physical_selection=physical_selection,
        max_vmax_kms=max_vmax_kms,
        extended=extended,
    )
    return paths


def test_process_case_selects_production_membership_and_brightest(tmp_path):
    paths = make_case(tmp_path)
    status = diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )
    assert status == "written"

    with paths.output_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == "2"
    assert row["physical_vmax_cut_applied"] == "false"
    assert row["physical_vmax_max_kms"] == ""
    assert row["population_selection"] == "full"
    assert row["retained_pointlike_count"] == "2"
    assert row["pointlike_npoint_metadata"] == "2"
    assert row["retained_extended_count"] == "2"
    assert row["max_halo_name"] == NAMES[2]
    assert row["max_hdf5_row"] == "2"
    assert float(row["max_Js"]) == 500.0
    assert float(row["max_theta_s_deg"]) == 2.0
    assert float(row["max_D_Earth_kpc"]) == 12.0
    assert row["max_classification"] == "extended"
    assert row["retained_catalog_path"] == str(paths.retained_catalog.resolve())

    with gzip.open(
        paths.retained_catalog, mode="rt", encoding="utf-8", newline=""
    ) as stream:
        retained = list(csv.DictReader(stream))
    assert [item["hdf5_row"] for item in retained] == ["0", "2", "4", "5"]
    assert [item["halo_name"] for item in retained] == [
        NAMES[0],
        NAMES[2],
        NAMES[4],
        NAMES[5],
    ]
    assert [item["component"] for item in retained] == [
        "pointlike",
        "extended",
        "extended",
        "pointlike",
    ]
    brightest_retained = retained[1]
    assert float(brightest_retained["Js"]) == float(row["max_Js"])
    assert float(brightest_retained["theta_s"]) == float(row["max_theta_s_deg"])
    assert float(brightest_retained["D_Earth"]) == float(row["max_D_Earth_kpc"])
    assert float(brightest_retained["Vmax"]) == 1.0
    assert float(brightest_retained["r_s"]) == 1.0
    assert float(brightest_retained["rho_s"]) == 2.0
    assert float(brightest_retained["Cv"]) == 1.0
    assert float(brightest_retained["Xearth"]) == -12.0
    assert float(brightest_retained["Yearth"]) == 0.0
    assert float(brightest_retained["Zearth"]) == 0.0

    assert diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=1,
        production_git_commit="production-commit",
        diagnostic_git_commit="another-commit",
    ) == "skipped"


def test_dark_selection_is_inclusive_and_excludes_bright_high_vmax(tmp_path):
    paths = make_case(
        tmp_path,
        physical_selection="selected",
        rows=dark_rows(),
        npoint=2,
    )
    diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )

    with paths.output_csv.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["physical_vmax_cut_applied"] == "true"
    assert float(row["physical_vmax_max_kms"]) == 8.0
    assert row["population_selection"] == "dark_vmax_le_8kms"
    assert row["retained_pointlike_count"] == "2"
    assert row["max_halo_name"] == NAMES[2]
    assert row["max_classification"] == "extended"
    assert float(row["max_Vmax_kms"]) == 8.0

    with gzip.open(
        paths.retained_catalog, mode="rt", encoding="utf-8", newline=""
    ) as stream:
        retained = list(csv.DictReader(stream))
    assert NAMES[0] not in {item["halo_name"] for item in retained}
    assert NAMES[1] in {item["halo_name"] for item in retained}
    assert all(float(item["Vmax"]) <= 8.0 for item in retained)


@pytest.mark.parametrize("bad_vmax", [np.nan, np.inf, 0.0, -1.0])
def test_dark_selection_rejects_invalid_vmax_for_otherwise_valid_halo(
    tmp_path, bad_vmax
):
    rows = dark_rows()
    rows[0][2] = bad_vmax
    paths = make_case(
        tmp_path,
        physical_selection="selected",
        rows=rows,
        npoint=2,
    )
    with pytest.raises(diagnostic.DiagnosticError, match="invalid Vmax"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_dark_selection_ignores_bad_vmax_on_otherwise_invalid_halo(tmp_path):
    rows = dark_rows()
    rows[3][2] = np.nan
    paths = make_case(
        tmp_path,
        physical_selection="selected",
        rows=rows,
        npoint=2,
    )
    assert diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    ) == "written"


def test_dark_selection_rejects_ineligible_extended_member(tmp_path):
    rows = dark_rows()
    rows[2][2] = 8.01
    paths = make_case(
        tmp_path,
        physical_selection="selected",
        rows=rows,
        npoint=2,
    )
    with pytest.raises(diagnostic.DiagnosticError, match="extended halo.*violates"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_process_case_reports_progress_and_completion(tmp_path, monkeypatch, capsys):
    paths = make_case(tmp_path)
    monkeypatch.setattr(diagnostic, "PROGRESS_INTERVAL_ROWS", 2)

    diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=1,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )

    assert capsys.readouterr().out.splitlines() == [
        "starting: fragile repop_0000",
        "progress: processed=2 rows, total=6 rows (33.3%)",
        "progress: processed=4 rows, total=6 rows (66.7%)",
        "progress: processed=6 rows, total=6 rows (100.0%)",
        "completed: fragile repop_0000: retained pointlike=2, extended=2",
    ]


def test_corrected_list_log_order_mismatch_fails_before_catalog_scan(tmp_path):
    paths = make_case(tmp_path, reverse_log=True)
    paths.catalog.unlink()
    with pytest.raises(diagnostic.DiagnosticError, match="identity mismatch"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_pointlike_npoint_mismatch_fails(tmp_path):
    paths = make_case(tmp_path, npoint=3)
    with pytest.raises(diagnostic.DiagnosticError, match="pointlike count mismatch"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=3,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )
    assert not paths.output_csv.exists()
    assert not paths.retained_catalog.exists()


def test_inconsistent_pointlike_fits_metadata_fails(tmp_path):
    paths = make_case(tmp_path)
    with fits.open(paths.pointlike_fits, mode="update") as hdul:
        hdul[2].header["JCPL"] = 101.0
    with pytest.raises(diagnostic.DiagnosticError, match="differs between map HDUs"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )
    assert not paths.output_csv.exists()
    assert not paths.retained_catalog.exists()


def test_explicit_full_population_metadata_is_supported(tmp_path):
    paths = make_case(tmp_path, physical_selection="full")
    diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )
    with paths.output_csv.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["physical_vmax_cut_applied"] == "false"
    assert row["population_selection"] == "full"


def test_selected_fits_and_list_ceiling_mismatch_fails(tmp_path):
    paths = make_case(
        tmp_path,
        physical_selection="selected",
        rows=dark_rows(),
        npoint=2,
    )
    write_extended_products(
        paths.corrected_list,
        paths.corrected_rendered_log,
        physical_selection="selected",
        max_vmax_kms=7.0,
    )
    with pytest.raises(diagnostic.DiagnosticError, match="ceiling disagrees"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_asymmetric_selection_metadata_fails(tmp_path):
    paths = make_case(
        tmp_path,
        physical_selection="selected",
        rows=dark_rows(),
        npoint=2,
    )
    write_extended_products(paths.corrected_list, paths.corrected_rendered_log)
    with pytest.raises(diagnostic.DiagnosticError, match="asymmetric"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


@pytest.mark.parametrize(
    ("vmax_cut", "vmax_max", "message"),
    [
        (True, None, "missing VMAXMAX"),
        (False, 8.0, "unexpected VMAXMAX"),
        (None, 8.0, "VMAXMAX without VMAXCUT"),
    ],
)
def test_invalid_fits_selection_metadata_fails(
    tmp_path, vmax_cut, vmax_max, message
):
    paths = make_case(tmp_path)
    with fits.open(paths.pointlike_fits, mode="update") as hdul:
        for hdu in hdul[1:3]:
            if vmax_cut is not None:
                hdu.header["VMAXCUT"] = vmax_cut
            if vmax_max is not None:
                hdu.header["VMAXMAX"] = vmax_max
    with pytest.raises(diagnostic.DiagnosticError, match=message):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_fits_selection_metadata_must_match_between_hdus(tmp_path):
    paths = make_case(tmp_path, physical_selection="full")
    with fits.open(paths.pointlike_fits, mode="update") as hdul:
        hdul[2].header["VMAXCUT"] = True
        hdul[2].header["VMAXMAX"] = 8.0
    with pytest.raises(diagnostic.DiagnosticError, match="VMAXCUT differs"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_selected_outputs_require_population_specific_directory(tmp_path):
    paths = make_case(
        tmp_path,
        physical_selection="selected",
        rows=dark_rows(),
        npoint=2,
    )
    unsafe_paths = diagnostic.CasePaths(
        catalog=paths.catalog,
        pointlike_fits=paths.pointlike_fits,
        corrected_list=paths.corrected_list,
        corrected_rendered_log=paths.corrected_rendered_log,
        retained_catalog=tmp_path / "results" / paths.retained_catalog.name,
        output_csv=tmp_path / "results" / paths.output_csv.name,
    )
    with pytest.raises(diagnostic.DiagnosticError, match="population tag"):
        diagnostic.process_case(
            paths=unsafe_paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


@pytest.mark.parametrize(
    ("metadata_override", "message"),
    [
        ({"scenario": "resilient"}, "scenario metadata mismatch"),
        ({"repop_id": 1}, "repop_id metadata mismatch"),
        ({"nside": 16}, "nside metadata mismatch"),
        ({"iteration": 1}, "HDF5 iteration mismatch"),
        (
            {
                "source_hdf5": (
                    "/production/repop_0000/fullrepop_hydro_resilient.h5"
                )
            },
            "source catalogue mismatch",
        ),
    ],
)
def test_corrected_list_case_provenance_mismatch_fails(
    tmp_path, metadata_override, message
):
    paths = make_case(tmp_path)
    write_extended_products(
        paths.corrected_list,
        paths.corrected_rendered_log,
        **metadata_override,
    )
    with pytest.raises(diagnostic.DiagnosticError, match=message):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_noncanonical_hdf5_row_identity_fails(tmp_path):
    paths = make_case(tmp_path)
    with h5py.File(paths.catalog, "r+") as handle:
        handle["iteration_0/halo_name"][1] = "hydro_res_99999"
    with pytest.raises(diagnostic.DiagnosticError, match="HDF5 identity mismatch at row 1"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_partial_checkpoint_is_regenerated_and_pair_is_restart_safe(tmp_path):
    paths = make_case(tmp_path)
    diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )
    paths.output_csv.unlink()
    assert paths.retained_catalog.exists()
    assert diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=3,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    ) == "written"
    assert paths.output_csv.exists()
    assert paths.retained_catalog.exists()


def test_checkpoint_selection_must_match_current_products(tmp_path):
    paths = make_case(
        tmp_path,
        physical_selection="selected",
        rows=dark_rows(),
        npoint=2,
    )
    diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )
    with paths.output_csv.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    row["population_selection"] = "full"
    with paths.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostic.OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(diagnostic.DiagnosticError, match="population_selection mismatch"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_schema_v1_checkpoint_is_not_accepted(tmp_path):
    paths = make_case(tmp_path)
    diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )
    with paths.output_csv.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    legacy_columns = tuple(
        column
        for column in diagnostic.OUTPUT_COLUMNS
        if column not in {
            "physical_vmax_cut_applied",
            "physical_vmax_max_kms",
            "population_selection",
        }
    )
    row["schema_version"] = "1"
    with paths.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=legacy_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(diagnostic.DiagnosticError, match="unexpected schema"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )


def test_failure_after_streaming_does_not_publish_outputs(tmp_path, monkeypatch):
    paths = make_case(tmp_path)
    original = diagnostic.scan_catalog

    def fail_after_scan(*args, **kwargs):
        original(*args, **kwargs)
        raise diagnostic.DiagnosticError("injected post-scan failure")

    monkeypatch.setattr(diagnostic, "scan_catalog", fail_after_scan)
    with pytest.raises(diagnostic.DiagnosticError, match="injected"):
        diagnostic.process_case(
            paths=paths,
            scenario="fragile",
            repop_id=0,
            nside=8,
            chunk_size=2,
            production_git_commit="production-commit",
            diagnostic_git_commit="diagnostic-commit",
        )
    assert not paths.output_csv.exists()
    assert not paths.retained_catalog.exists()


def test_merge_case_outputs_is_deterministic(tmp_path):
    paths = make_case(tmp_path)
    diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )
    merged = tmp_path / "merged.csv"
    diagnostic.merge_case_outputs([paths.output_csv], merged)
    assert merged.read_text(encoding="utf-8") == paths.output_csv.read_text(
        encoding="utf-8"
    )


def test_merge_rejects_mixed_population_selections(tmp_path):
    paths = make_case(tmp_path)
    diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )
    with paths.output_csv.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    row["repop_id"] = "1"
    row["physical_vmax_cut_applied"] = "true"
    row["physical_vmax_max_kms"] = "8"
    row["population_selection"] = "dark_vmax_le_8kms"
    selected_csv = tmp_path / "selected.csv"
    with selected_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostic.OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(diagnostic.DiagnosticError, match="mixed physical"):
        diagnostic.merge_case_outputs(
            [paths.output_csv, selected_csv],
            tmp_path / "must-not-exist.csv",
        )
    assert not (tmp_path / "must-not-exist.csv").exists()


def test_merge_rejects_incompatible_selected_vmax_ceilings(tmp_path):
    paths = make_case(tmp_path)
    diagnostic.process_case(
        paths=paths,
        scenario="fragile",
        repop_id=0,
        nside=8,
        chunk_size=2,
        production_git_commit="production-commit",
        diagnostic_git_commit="diagnostic-commit",
    )
    with paths.output_csv.open(newline="", encoding="utf-8") as stream:
        base = next(csv.DictReader(stream))

    selected_paths = []
    for repop_id, ceiling in ((0, "8"), (1, "7")):
        row = dict(base)
        row["repop_id"] = str(repop_id)
        row["physical_vmax_cut_applied"] = "true"
        row["physical_vmax_max_kms"] = ceiling
        row["population_selection"] = f"dark_vmax_le_{ceiling}kms"
        path = tmp_path / f"selected_{ceiling}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=diagnostic.OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
        selected_paths.append(path)

    with pytest.raises(diagnostic.DiagnosticError, match="mixed physical"):
        diagnostic.merge_case_outputs(
            selected_paths,
            tmp_path / "must-not-exist.csv",
        )
