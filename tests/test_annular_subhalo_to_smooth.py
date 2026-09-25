import csv
import importlib.util
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pytest
from astropy.io import fits


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCAN_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "maps"
    / "scripts"
    / "scan_annular_subhalo_to_smooth.py"
)
PLOT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "maps"
    / "scripts"
    / "plot_annular_subhalo_to_smooth.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scan = load_module("scan_annular_subhalo_to_smooth", SCAN_PATH)
plot = load_module("plot_annular_subhalo_to_smooth", PLOT_PATH)


def write_final_map(
    path,
    smooth,
    pointlike,
    extended,
    nside,
    ordering="NESTED",
    unit="GeV^2 cm^-5",
):
    pixels = np.arange(hp.nside2npix(nside), dtype=np.int32)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PIXEL", format="1J", array=pixels),
            fits.Column(
                name="Jsmooth",
                format="1D",
                unit=unit,
                array=np.asarray(smooth, dtype=np.float64),
            ),
            fits.Column(
                name="Jpointlike",
                format="1D",
                unit=unit,
                array=np.asarray(pointlike, dtype=np.float64),
            ),
            fits.Column(
                name="Jextended",
                format="1D",
                unit=unit,
                array=np.asarray(extended, dtype=np.float64),
            ),
        ],
        name="JFACTOR",
    )
    hdu.header["PIXTYPE"] = "HEALPIX"
    hdu.header["ORDERING"] = ordering
    hdu.header["NSIDE"] = nside
    hdu.header["INDXSCHM"] = "EXPLICIT"
    hdu.header["COORDSYS"] = "G"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


def write_scenario_map(tmp_path, prefix, smooth, pointlike, extended, nside=2):
    path = tmp_path / f"{prefix}.fits"
    write_final_map(path, smooth, pointlike, extended, nside)
    return path


def test_pixels_are_assigned_to_correct_gc_annuli(tmp_path):
    nside = 2
    npix = hp.nside2npix(nside)
    smooth = np.ones(npix)
    pointlike = np.full(npix, 0.2)
    extended = np.full(npix, 0.1)
    path = write_scenario_map(
        tmp_path,
        "assignment",
        smooth,
        pointlike,
        extended,
        nside,
    )

    records = scan.scan_all(
        {"fragile": path, "resilient": path},
        repop_id=7,
        bin_width_deg=60.0,
        chunk_size=5,
    )
    fragile = [row for row in records if row["scenario"] == "fragile"]

    pixels = np.arange(npix)
    lon_deg, lat_deg = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
    cos_psi = np.cos(np.radians(lat_deg)) * np.cos(np.radians(lon_deg))
    psi_deg = np.degrees(np.arccos(np.clip(cos_psi, -1.0, 1.0)))
    expected_counts, _ = np.histogram(psi_deg, bins=[0.0, 60.0, 120.0, 180.0])

    assert [row["n_pixels"] for row in fragile] == expected_counts.tolist()
    assert sum(row["n_pixels"] for row in fragile) == npix


def test_ratio_uses_ratio_of_sums_and_adds_both_subhalo_components(tmp_path):
    nside = 1
    npix = hp.nside2npix(nside)
    smooth = np.ones(npix)
    smooth[1] = 100.0
    pointlike = np.ones(npix)
    pointlike[0] = 100.0
    pointlike[1] = 0.0
    extended = np.full(npix, 0.25)
    path = write_scenario_map(
        tmp_path,
        "ratio",
        smooth,
        pointlike,
        extended,
        nside,
    )

    records = scan.scan_all(
        {"fragile": path, "resilient": path},
        repop_id=3,
        bin_width_deg=180.0,
        chunk_size=4,
    )
    row = records[0]
    expected_pointlike = pointlike.sum()
    expected_extended = extended.sum()
    expected_sub = expected_pointlike + expected_extended
    expected_ratio = expected_sub / smooth.sum()
    simple_pixel_ratio_mean = np.mean((pointlike + extended) / smooth)

    assert row["sum_j_pointlike"] == pytest.approx(expected_pointlike)
    assert row["sum_j_extended"] == pytest.approx(expected_extended)
    assert row["sum_j_sub"] == pytest.approx(expected_sub)
    assert row["ratio_sub_to_smooth"] == pytest.approx(expected_ratio)
    assert row["ratio_sub_to_smooth"] != pytest.approx(simple_pixel_ratio_mean)


@pytest.mark.parametrize("mismatch", ["nside", "ordering"])
def test_incompatible_nside_or_ordering_is_rejected(tmp_path, mismatch):
    fragile_path = tmp_path / "fragile.fits"
    resilient_path = tmp_path / "resilient.fits"
    write_final_map(
        fragile_path,
        np.ones(12),
        np.ones(12),
        np.ones(12),
        nside=1,
    )

    if mismatch == "nside":
        write_final_map(
            resilient_path,
            np.ones(48),
            np.ones(48),
            np.ones(48),
            nside=2,
        )
        message = "NSIDE"
    else:
        write_final_map(
            resilient_path,
            np.ones(12),
            np.ones(12),
            np.ones(12),
            nside=1,
            ordering="RING",
        )
        message = "ORDERING"

    with pytest.raises(ValueError, match=message):
        scan.scan_all(
            {"fragile": fragile_path, "resilient": resilient_path},
            repop_id=0,
            bin_width_deg=180.0,
            chunk_size=10,
        )


def test_each_scenario_fits_is_opened_once(tmp_path, monkeypatch):
    values = np.ones(12)
    fragile_path = write_scenario_map(
        tmp_path,
        "fragile_once",
        values,
        values,
        values,
        nside=1,
    )
    resilient_path = write_scenario_map(
        tmp_path,
        "resilient_once",
        values,
        values,
        values,
        nside=1,
    )
    opened = []
    original_open = scan.fits.open

    def tracked_open(path, *args, **kwargs):
        opened.append(Path(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(scan.fits, "open", tracked_open)
    scan.scan_all(
        {"fragile": fragile_path, "resilient": resilient_path},
        repop_id=0,
        bin_width_deg=180.0,
        chunk_size=5,
    )

    assert opened.count(fragile_path.resolve()) == 1
    assert opened.count(resilient_path.resolve()) == 1


def test_plot_only_reuses_csv_without_fits(tmp_path, monkeypatch):
    csv_path = tmp_path / "saved_scan.csv"
    rows = []
    for scenario, ratio in (("fragile", 0.2), ("resilient", 0.4)):
        rows.append(
            {
                "scenario": scenario,
                "repop_id": 11,
                "psi_min_deg": 0.0,
                "psi_max_deg": 180.0,
                "psi_center_deg": 90.0,
                "n_pixels": 12,
                "sum_j_smooth": 10.0,
                "sum_j_pointlike": 10.0 * ratio * 0.75,
                "sum_j_extended": 10.0 * ratio * 0.25,
                "sum_j_sub": 10.0 * ratio,
                "ratio_sub_to_smooth": ratio,
            }
        )
    scan.write_records(csv_path, rows)

    output_dir = tmp_path / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PLOT_PATH),
            "--input-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    plot.main()

    assert (output_dir / "saved_scan.png").is_file()
    assert (output_dir / "saved_scan.pdf").is_file()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 2
