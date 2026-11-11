import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pytest
from astropy.io import fits


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import prepare_subhalo_components as prep


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


def write_catalogue(path, rows, names):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("iteration_0")
        data = group.create_dataset("data", data=np.asarray(rows, dtype=float))
        data.attrs["column_names"] = COLUMNS
        group.create_dataset(
            "halo_name",
            data=np.asarray([name.encode() for name in names]),
        )


def run_preparation(tmp_path, rows, names, *, max_vmax_km_s):
    input_h5 = tmp_path / "catalogue.h5"
    output_list = tmp_path / "extended.txt"
    output_fits = tmp_path / "pointlike.fits"
    write_catalogue(input_h5, rows, names)
    prep.prepare_subhalo_components(
        input_h5=input_h5,
        output_list=output_list,
        output_pointlike_fits=output_fits,
        scenario="fragile",
        repop_id=0,
        iteration=0,
        top_n=None,
        halo_type="DSPH",
        nside=8,
        round_up_decimals=2,
        chunk_size=2,
        extended_cut_f=1.0e-3,
        pointlike_cut_f=1.0e-3,
        max_vmax_km_s=max_vmax_km_s,
    )
    return output_list, output_fits


def component_names(output_list):
    return [
        line.split()[0]
        for line in output_list.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def test_dark_selection_precedes_jref_and_component_cuts(tmp_path):
    nside = 8
    theta_min = prep.round_up(prep.healpix_pixel_size_deg(nside), 2)
    ext_factor = prep.clumpy_central_pixel_proxy_from_js(
        js=np.array([1.0]),
        d_earth_kpc=np.array([10.0]),
        rs_kpc=np.array([1.0]),
        nside=nside,
    )[0]
    rows = [
        # These bright Vmax>8 objects must neither define J_ref nor survive.
        [100.0, 10.0, 9.0, 1.0, 1.0, 1.0, 1e7, 10.0, 0.0, 0.0],
        [200.0 / ext_factor, 10.0, 12.0, theta_min + 1.0, 1.0, 1.0, 1e7, 0.0, 0.0, 10.0],
        # Vmax=8 is inclusive and defines dark J_ref=50.
        [50.0, 10.0, 8.0, 1.0, 1.0, 1.0, 1e7, 0.0, 10.0, 0.0],
        # Dark objects immediately above and below the 0.05 thresholds.
        [0.06, 10.0, 4.0, 1.0, 1.0, 1.0, 1e7, -10.0, 0.0, 0.0],
        [0.04, 10.0, 4.0, 1.0, 1.0, 1.0, 1e7, 0.0, -10.0, 0.0],
        [0.06 / ext_factor, 10.0, 4.0, theta_min + 1.0, 1.0, 1.0, 1e7, 0.0, 0.0, -10.0],
        [0.04 / ext_factor, 10.0, 4.0, theta_min + 1.0, 1.0, 1.0, 1e7, 1.0, 1.0, 1.0],
    ]
    names = [
        "dwarf_point",
        "dwarf_extended",
        "boundary_point",
        "dark_point_keep",
        "dark_point_drop",
        "dark_extended_keep",
        "dark_extended_drop",
    ]

    output_list, output_fits = run_preparation(
        tmp_path, rows, names, max_vmax_km_s=8.0
    )

    assert component_names(output_list) == ["dark_extended_keep"]
    text = output_list.read_text()
    assert "Physical Vmax selection: Vmax <= 8.0 km/s" in text
    assert "# J_pixel_ref = 5.00000000e+01" in text
    assert "# extended_J_cut = 0.05" in text
    assert "# pointlike_J_cut = 0.05" in text

    with fits.open(output_fits) as hdul:
        header = hdul["JFACTOR"].header
        pointlike = np.asarray(hdul["JFACTOR"].data["Jpointlike"])
        assert header["VMAXCUT"] is True
        assert header["VMAXMAX"] == pytest.approx(8.0)
        assert header["JREF"] == pytest.approx(50.0)
        assert header["JCPL"] == pytest.approx(0.05)
        assert header["JCEXT"] == pytest.approx(0.05)
        assert header["NPOINT"] == 2
        assert pointlike.sum() == pytest.approx(50.06)
        dwarf_pixel = hp.ang2pix(
            nside, 0.0, 0.0, lonlat=True, nest=True
        )
        assert pointlike[dwarf_pixel] == 0.0


def test_disabled_selection_preserves_full_population(tmp_path):
    theta_min = prep.round_up(prep.healpix_pixel_size_deg(8), 2)
    rows = [
        [10.0, 10.0, 9.0, 1.0, 1.0, 1.0, 1e7, 10.0, 0.0, 0.0],
        [20.0, 10.0, 12.0, theta_min + 1.0, 1.0, 1.0, 1e7, 0.0, 10.0, 0.0],
    ]
    output_list, output_fits = run_preparation(
        tmp_path, rows, ["full_point", "full_extended"], max_vmax_km_s=None
    )
    assert component_names(output_list) == ["full_extended"]
    with fits.open(output_fits) as hdul:
        assert hdul["JFACTOR"].header["VMAXCUT"] is False
        assert hdul["JFACTOR"].header["NPTOTAL"] == 1
        assert np.asarray(hdul["JFACTOR"].data["Jpointlike"]).sum() == pytest.approx(10.0)

    # Legacy full-population FITS predate VMAXCUT and must remain reusable.
    with fits.open(output_fits, mode="update") as hdul:
        jref = float(hdul["JFACTOR"].header["JREF"])
        for hdu in (hdul["JFACTOR"], hdul["JFACTOR_PER_SR"]):
            del hdu.header["VMAXCUT"]
    valid, reason = prep.validate_existing_pointlike_fits(
        output_fits,
        nside=8,
        scenario="fragile",
        repop_id=0,
        theta_cut_deg=theta_min,
        pointlike_cut_f=1.0e-3,
        j_pixel_ref=jref,
        max_vmax_km_s=None,
    )
    assert valid, reason


def test_output_namespaces_and_fits_reuse_cannot_mix_populations(tmp_path):
    full_root = prep.get_clumpy_root(None)
    dark_root = prep.get_clumpy_root(8.0)
    assert full_root == prep.BASE_RUN_DIR
    assert dark_root == prep.BASE_RUN_DIR / "dark_vmax_le_8kms"
    assert prep.get_output_pointlike_fits(0, "fragile", 2048).is_relative_to(full_root)
    assert prep.get_output_pointlike_fits(
        0, "fragile", 2048, max_vmax_km_s=8.0
    ).is_relative_to(dark_root)
    with pytest.raises(ValueError, match="selection namespace"):
        prep.get_clumpy_root(8.0, tmp_path / "full_population")
    inplace_root = tmp_path / "full_population"
    assert prep.get_clumpy_root(
        8.0,
        inplace_root,
        allow_inplace_selected_output=True,
    ) == inplace_root
    assert prep.get_output_pointlike_fits(
        0,
        "fragile",
        2048,
        max_vmax_km_s=8.0,
        clumpy_root=inplace_root,
        allow_inplace_selected_output=True,
    ).is_relative_to(inplace_root)

    rows = [[10.0, 10.0, 9.0, 1.0, 1.0, 1.0, 1e7, 10.0, 0.0, 0.0]]
    _, full_fits = run_preparation(
        tmp_path, rows, ["full_point"], max_vmax_km_s=None
    )
    valid, reason = prep.validate_existing_pointlike_fits(
        full_fits,
        nside=8,
        scenario="fragile",
        repop_id=0,
        theta_cut_deg=prep.round_up(prep.healpix_pixel_size_deg(8), 2),
        pointlike_cut_f=1.0e-3,
        j_pixel_ref=10.0,
        max_vmax_km_s=8.0,
    )
    assert valid is False
    assert "VMAXCUT" in reason
