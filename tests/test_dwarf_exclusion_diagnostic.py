import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "dwarf_exclusion"
    / "scripts"
    / "analyze_dwarf_exclusion.py"
)
spec = importlib.util.spec_from_file_location("dwarf_exclusion", SCRIPT)
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)


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
    "Msun / kpc3",
    "kpc",
    "kpc",
    "kpc",
]


def write_catalogue(path, rows):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("iteration_0")
        data = group.create_dataset("data", data=np.asarray(rows, dtype=np.float64))
        data.attrs["column_names"] = COLUMNS
        data.attrs["units"] = UNITS
        group.create_dataset(
            "halo_name",
            data=np.asarray([f"hydro_res_{index + 1:05d}".encode() for index in range(len(rows))]),
        )


def test_selection_matches_production_reference_and_invariants(tmp_path):
    nside = 8
    theta_min = diag.round_up(diag.healpix_pixel_size_deg(nside), 2)
    # Choose an extended Js whose central-pixel proxy lies between the two cuts.
    factor = diag.clumpy_central_pixel_proxy_from_js(
        js=np.array([1.0]), d_earth_kpc=np.array([10.0]), rs_kpc=np.array([1.0]), nside=nside
    )[0]
    extended_new_js = 0.075 / factor
    rows = [
        # Original reference: point-like dwarf, proxy=Js=100.
        [100.0, 10.0, 12.0, 1.0, 1.0, 1.0, 1e7, 10.0, 0.0, 0.0],
        # Dark-only reference: point-like, proxy=50.
        [50.0, 10.0, 8.0, 1.0, 1.0, 1.0, 1e7, 0.0, 10.0, 0.0],
        # Newly admitted point-like: 0.05 <= Js=0.075 < 0.1.
        [0.075, 10.0, 4.0, 1.0, 1.0, 1.0, 1e7, -10.0, 0.0, 0.0],
        # Newly admitted extended with proxy=0.075.
        [extended_new_js, 10.0, 4.0, theta_min + 1.0, 1.0, 1.0, 1e7, 0.0, -10.0, 0.0],
        # Extended dwarf, below the original cut.
        [0.01 / factor, 10.0, 9.0, theta_min + 1.0, 1.0, 1.0, 1e7, 0.0, 0.0, 10.0],
    ]
    path = tmp_path / "catalogue.h5"
    write_catalogue(path, rows)

    result = diag.scan_catalogue(
        path,
        repop_id=0,
        scenario="fragile",
        nside=nside,
        chunk_size=2,
        progress=False,
    )

    # Independently call the production reference routine on the same HDF5 table.
    with h5py.File(path, "r") as handle:
        data = handle["iteration_0/data"]
        indices = {name: COLUMNS.index(name) for name in diag.REQUIRED_COLUMNS}
        production = __import__("prepare_subhalo_components").compute_pixel_reference(
            data=data,
            column_indices=indices,
            n_total=len(rows),
            theta_min_deg=theta_min,
            nside=nside,
            chunk_size=2,
        )

    assert result["jref_original"] == production["j_pixel_ref"] == 100.0
    assert result["jref_dark"] == 50.0
    assert result["jref_ratio_dark_to_original"] == 0.5
    assert result["original_ref_is_dwarf"] is True
    assert result["n_dwarfs_total"] == 2
    assert result["n_dwarfs_pointlike"] == 1
    assert result["n_dwarfs_extended"] == 1
    assert result["n_newly_admitted_pointlike"] == 1
    assert result["n_newly_admitted_extended"] == 1
    assert result["n_a_not_b_total"] == 0
    assert result["selection_b_is_superset"] is True
    assert np.isclose(result["newly_admitted_js_sum_pointlike"], 0.075)
    assert np.isclose(result["newly_admitted_js_sum_extended"], extended_new_js)

    # Run the production preparation implementation and compare its original
    # retained counts with diagnostic A (dark retained + retained dwarfs).
    output_list = tmp_path / "production_extended.txt"
    output_fits = tmp_path / "production_pointlike.fits"
    __import__("prepare_subhalo_components").prepare_subhalo_components(
        input_h5=path,
        output_list=output_list,
        output_pointlike_fits=output_fits,
        scenario="fragile",
        repop_id=0,
        iteration=0,
        top_n=None,
        halo_type="DSPH",
        nside=nside,
        round_up_decimals=2,
        chunk_size=2,
        extended_cut_f=diag.EXTENDED_F,
        pointlike_cut_f=diag.POINTLIKE_F,
    )
    production_extended_count = sum(
        1
        for line in output_list.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    with fits.open(output_fits) as hdul:
        production_pointlike_count = int(hdul["JFACTOR"].header["NPOINT"])
    assert production_pointlike_count == (
        result["n_dark_retained_a_pointlike"]
        + result["n_dwarfs_passing_original_pointlike"]
    )
    assert production_extended_count == (
        result["n_dark_retained_a_extended"]
        + result["n_dwarfs_passing_original_extended"]
    )


def test_equal_reference_produces_identical_dark_selection(tmp_path):
    rows = [
        [20.0, 10.0, 8.0, 1.0, 1.0, 1.0, 1e7, 10.0, 0.0, 0.0],
        [10.0, 10.0, 12.0, 1.0, 1.0, 1.0, 1e7, 0.0, 10.0, 0.0],
        [0.03, 10.0, 4.0, 1.0, 1.0, 1.0, 1e7, -10.0, 0.0, 0.0],
    ]
    path = tmp_path / "catalogue.h5"
    write_catalogue(path, rows)
    result = diag.scan_catalogue(
        path, repop_id=1, scenario="resilient", nside=8, chunk_size=1, progress=False
    )
    assert result["jref_dark"] == result["jref_original"]
    assert result["n_newly_admitted_total"] == 0
    assert result["n_a_not_b_total"] == 0
