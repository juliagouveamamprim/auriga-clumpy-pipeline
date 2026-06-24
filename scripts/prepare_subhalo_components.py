#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split an Auriga full-repopulation HDF5 catalog into extended and
pointlike components in a single chunked pass.

Usage
-----

    python3 prepare_subhalo_components.py 1 resilient
    python3 prepare_subhalo_components.py 1 fragile

This reads:

    outputs/repop_0001/fullrepop_hydro_<scenario>.h5

and writes:

    outputs/clumpy/<scenario>/lists/raw/
        repop_0001_raw_nopointlike.txt

    outputs/clumpy/<scenario>/pointlike/
        repop_0001_pointlike_nside1024.fits

Expected HDF5 structure
-----------------------

    iteration_0/data
    iteration_0/halo_name

where data is a two-dimensional table whose column names are stored
in the ``column_names`` HDF5 attribute.

The required columns are:

    Js, D_Earth, theta_s, r_s, rho_s, Xearth, Yearth, Zearth

The CLUMPY list uses:

    Name Type l b d z Rdelta rhos rs prof #1 #2 #3

For NFW:

    rho(rs) = rho_s / 4

Operational choices:

    Type   = DSPH
    prof   = kZHAO
    params = alpha,beta,gamma = 1,3,1
    Rdelta = r_s
"""

import argparse
import math
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
from astropy.io import fits


# ============================================================
# GLOBAL CONFIG
# ============================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_H5_DIR = REPOSITORY_ROOT / "outputs"
BASE_RUN_DIR = REPOSITORY_ROOT / "outputs" / "clumpy"

# Internal HDF5 iteration index.
# In our one-directory-per-repop convention, each HDF5 contains iteration_0.
ITERATION = 0

# For tests:
#   TOP_N = 5000
#
# For the full list:
#   TOP_N = None
#
# Important: TOP_N is applied AFTER the non-point-like cut.
TOP_N = None

# CLUMPY halo type.
HALO_TYPE = "DSPH"

# HEALPix map resolution used to define the point-like cut.
NSIDE = 1024

# Number of decimal places used when rounding theta_pix upward.
# Example: if theta_pix = 0.05726 deg and ROUND_UP_DECIMALS = 2,
# then theta_min_deg = 0.06 deg.
ROUND_UP_DECIMALS = 2

# Number of HDF5 rows read at a time.
#
# 500_000 is conservative for both the legacy 14-column schema
# and the reduced 10-column schema, plus names and temporary arrays.
CHUNK_SIZE = 500_000


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert one Auriga HDF5 full repopulation into a CLUMPY "
            "raw halo list without point-like halos."
        )
    )

    parser.add_argument(
        "repop_id",
        type=int,
        help=(
            "Global repopulation ID. Example: 1 reads "
            "outputs/repop_0001/ inside the repository."
        ),
    )

    parser.add_argument(
        "scenario",
        choices=["resilient", "fragile"],
        help="Hydro scenario to process.",
    )

    return parser.parse_args()


# ============================================================
# Path helpers
# ============================================================

def get_h5_dir(repop_id):
    """
    Return HDF5 directory for a given global repop ID.

    All repops are expected to follow:

        outputs/repop_XXXX/
    """
    return BASE_H5_DIR / f"repop_{repop_id:04d}"


def get_input_h5(repop_id, scenario):
    """
    Return input HDF5 file for one repop/scenario.
    """
    return get_h5_dir(repop_id) / f"fullrepop_hydro_{scenario}.h5"


def get_output_list(repop_id, scenario, top_n):
    """
    Return output CLUMPY raw list path.
    """
    if top_n is None:
        filename = f"repop_{repop_id:04d}_raw_nopointlike.txt"
    else:
        filename = f"repop_{repop_id:04d}_raw_nopointlike_top{top_n}.txt"

    return (
        BASE_RUN_DIR
        / scenario
        / "lists"
        / "raw"
        / filename
    )


def get_output_pointlike_fits(repop_id, scenario, nside):
    """Return output path for the pointlike-only HEALPix FITS map."""
    return (
        BASE_RUN_DIR
        / scenario
        / "pointlike"
        / f"repop_{repop_id:04d}_pointlike_nside{nside}.fits"
    )


# ============================================================
# Geometry / filtering helpers
# ============================================================

def healpix_pixel_size_deg(nside):
    """
    Equivalent angular size of a HEALPix pixel in degrees:

        theta_pix = sqrt(Omega_pix)

    with:

        Omega_pix = 4*pi / (12*nside^2)
    """
    omega_pix_sr = 4.0 * math.pi / (12.0 * nside * nside)
    theta_pix_rad = math.sqrt(omega_pix_sr)
    return math.degrees(theta_pix_rad)


def round_up(value, decimals=2):
    """
    Round upward to a fixed number of decimal places.
    """
    factor = 10 ** decimals
    return math.ceil(value * factor) / factor


def xyz_to_lb_deg(x, y, z):
    """
    Convert Earth-centered Cartesian Galactic coordinates to (l, b) in degrees.

    Convention assumed:
      +X points toward the Galactic Center
      +Y points toward l = +90 deg
      +Z points toward the North Galactic Pole
    """
    r = np.sqrt(x**2 + y**2 + z**2)

    l = np.degrees(np.arctan2(y, x)) % 360.0
    b = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))

    return l, b


def decode_halo_names(names):
    """
    Decode HDF5 halo names, which may be stored as bytes.
    """
    decoded = []

    for name in names:
        if isinstance(name, bytes):
            decoded.append(name.decode("utf-8"))
        else:
            decoded.append(str(name))

    return decoded


# ============================================================
# Output helpers
# ============================================================

def write_header(
    f,
    input_h5,
    output_list,
    scenario,
    repop_id,
    iteration,
    nside,
    theta_pix_deg,
    theta_min_deg,
    chunk_size,
    halo_type,
    top_n,
):
    """
    Write CLUMPY list header.
    """

    header = [
        "#************************************************************************************************************",
        "# Custom CLUMPY halo list generated from Auriga full-repopulation HDF5",
        f"# Source HDF5: {input_h5}",
        f"# Output list: {output_list}",
        f"# Scenario: {scenario}",
        f"# REPOP_ID: {repop_id}",
        f"# HDF5 internal iteration: {iteration}",
        "#",
        "# This list keeps only extended/non-point-like subhalos.",
        f"# Point-like filter: theta_s >= {theta_min_deg:.6f} deg",
        f"# HEALPix NSIDE: {nside}",
        f"# Equivalent HEALPix pixel size: {theta_pix_deg:.6f} deg",
        f"# Conservative threshold: {theta_min_deg:.6f} deg",
        f"# Chunk size used while reading HDF5: {chunk_size}",
        f"# TOP_N after non-point-like cut: {top_n}",
        "#",
        "# HDF5 columns are identified by the column_names attribute.",
        "# Required columns: Js, D_Earth, theta_s, r_s, rho_s,",
        "#                   Xearth, Yearth, Zearth.",
        "#",
        "# CLUMPY conversion notes:",
        f"# - Halo type set operationally to {halo_type}.",
        "# - NFW implemented as kZHAO with (alpha, beta, gamma) = (1, 3, 1).",
        "# - Rdelta = r_s, so each halo is truncated at r_s.",
        "# - rhos below is rho(rs) = rho_s / 4 for NFW.",
        "# - The HDF5 order is preserved; no re-sorting is applied.",
        "#",
        "# Format:",
        "# Name  Type  l  b  d  z  Rdelta  rhos  rs  prof  #1  #2  #3",
        "#************************************************************************************************************",
        "# Name           Type      l[deg]      b[deg]      d[kpc]   z      Rdelta[kpc]   rhos[Msun/kpc3]   rs[kpc]   prof.   #1   #2   #3",
    ]

    for line in header:
        f.write(line + "\n")


def write_rows_for_chunk(
    f,
    names,
    arr,
    mask,
    halo_type,
    column_indices,
):
    """
    Write CLUMPY rows for one filtered chunk.

    Columns are selected by name through ``column_indices``.
    """

    if not np.any(mask):
        return 0

    selected = arr[mask]
    selected_names = [name for name, keep in zip(names, mask) if keep]

    d_earth = selected[:, column_indices["D_Earth"]]
    r_s = selected[:, column_indices["r_s"]]
    rho_s_scale = selected[:, column_indices["rho_s"]]

    x_e = selected[:, column_indices["Xearth"]]
    y_e = selected[:, column_indices["Yearth"]]
    z_e = selected[:, column_indices["Zearth"]]

    l_deg, b_deg = xyz_to_lb_deg(x_e, y_e, z_e)

    # For NFW, CLUMPY wants rho(rs), while the HDF5 stores rho_s.
    rhos_clumpy = rho_s_scale / 4.0

    n_written = 0

    for i, name in enumerate(selected_names):
        f.write(
            f"{name:<15s} "
            f"{halo_type:<8s} "
            f"{l_deg[i]:>10.6f} "
            f"{b_deg[i]:>10.6f} "
            f"{d_earth[i]:>12.6f} "
            f"{-1:>6.0f} "
            f"{r_s[i]:>12.6f} "
            f"{rhos_clumpy[i]:>16.8e} "
            f"{r_s[i]:>10.6f} "
            f"{'kZHAO':<8s} "
            f"{1:>4.0f} "
            f"{3:>4.0f} "
            f"{1:>4.0f}\n"
        )

        n_written += 1

    return n_written


def write_pointlike_fits(
    output_path,
    pointlike_map,
    nside,
    scenario,
    repop_id,
    theta_cut_deg,
    n_pointlike,
):
    """Write the pointlike component as a CLUMPY-compatible HEALPix FITS."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npix = hp.nside2npix(nside)
    pixel_area_sr = hp.nside2pixarea(nside)

    pointlike_map = np.asarray(pointlike_map, dtype=np.float64)

    if pointlike_map.shape != (npix,):
        raise ValueError(
            f"Expected pointlike map shape {(npix,)}, "
            f"got {pointlike_map.shape}."
        )

    if not np.all(np.isfinite(pointlike_map)):
        raise ValueError("Pointlike map contains non-finite values.")

    if np.any(pointlike_map < 0.0):
        raise ValueError("Pointlike map contains negative J-factor values.")

    pixels = np.arange(npix, dtype=np.int32)
    pointlike_per_sr = pointlike_map / pixel_area_sr

    hdu_j = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="PIXEL",
                format="1J",
                array=pixels,
            ),
            fits.Column(
                name="Jpointlike",
                format="1D",
                unit="GeV^2 cm^-5",
                array=pointlike_map,
            ),
        ],
        name="JFACTOR",
    )

    hdu_per_sr = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="PIXEL",
                format="1J",
                array=pixels,
            ),
            fits.Column(
                name="Jpointlike_per_sr",
                format="1D",
                unit="GeV^2 cm^-5 sr^-1",
                array=pointlike_per_sr,
            ),
        ],
        name="JFACTOR_PER_SR",
    )

    for hdu in (hdu_j, hdu_per_sr):
        hdu.header["PIXTYPE"] = "HEALPIX"
        hdu.header["ORDERING"] = "NESTED"
        hdu.header["NSIDE"] = nside
        hdu.header["FIRSTPIX"] = 0
        hdu.header["LASTPIX"] = npix - 1
        hdu.header["INDXSCHM"] = "EXPLICIT"
        hdu.header["COORDSYS"] = "G"
        hdu.header["OBJECT"] = "FULLSKY"
        hdu.header["SCENARIO"] = scenario
        hdu.header["REPOPID"] = repop_id
        hdu.header["THETACUT"] = (
            theta_cut_deg,
            "Pointlike cut: theta_s < THETACUT [deg]",
        )
        hdu.header["NPOINT"] = (
            n_pointlike,
            "Number of pointlike subhalos",
        )
        hdu.header["PIXAREA"] = (
            pixel_area_sr,
            "HEALPix pixel solid angle [sr]",
        )

    primary = fits.PrimaryHDU()
    primary.header["CONTENT"] = "Auriga pointlike subhalo J-factor map"

    fits.HDUList(
        [
            primary,
            hdu_j,
            hdu_per_sr,
        ]
    ).writeto(output_path, overwrite=True)


# ============================================================
# Main conversion
# ============================================================

def prepare_subhalo_components(
    input_h5,
    output_list,
    output_pointlike_fits,
    scenario,
    repop_id,
    iteration,
    top_n,
    halo_type,
    nside,
    round_up_decimals,
    chunk_size,
):
    """
    Build the extended CLUMPY list and pointlike HEALPix map
    in a single chunked pass through the HDF5 catalog.
    """

    input_h5 = Path(input_h5)
    output_list = Path(output_list)
    output_pointlike_fits = Path(output_pointlike_fits)

    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_pointlike_fits.parent.mkdir(parents=True, exist_ok=True)

    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_h5}")

    theta_pix_deg = healpix_pixel_size_deg(nside)
    theta_min_deg = round_up(theta_pix_deg, decimals=round_up_decimals)

    group_name = f"iteration_{iteration}"

    pointlike_map = np.zeros(
        hp.nside2npix(nside),
        dtype=np.float64,
    )

    total_seen = 0
    total_valid = 0
    total_invalid = 0

    total_extended = 0
    total_extended_written = 0

    total_pointlike = 0
    total_pointlike_js = 0.0

    with h5py.File(input_h5, "r") as h5:
        if group_name not in h5:
            raise KeyError(f"Could not find group '{group_name}' in {input_h5}")

        group = h5[group_name]

        if "data" not in group:
            raise KeyError(f"Could not find dataset '{group_name}/data'")

        if "halo_name" not in group:
            raise KeyError(f"Could not find dataset '{group_name}/halo_name'")

        data = group["data"]
        halo_name = group["halo_name"]

        if data.ndim != 2:
            raise ValueError(
                f"Expected '{group_name}/data' to be two-dimensional, "
                f"got shape {data.shape}."
            )

        if "column_names" not in data.attrs:
            raise KeyError(
                f"Dataset '{group_name}/data' has no column_names attribute."
            )

        column_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in data.attrs["column_names"]
        ]

        if len(column_names) != data.shape[1]:
            raise ValueError(
                f"column_names contains {len(column_names)} entries, but "
                f"'{group_name}/data' has {data.shape[1]} columns."
            )

        required_columns = [
            "Js",
            "D_Earth",
            "theta_s",
            "r_s",
            "rho_s",
            "Xearth",
            "Yearth",
            "Zearth",
        ]

        missing_columns = [
            name for name in required_columns
            if name not in column_names
        ]

        if missing_columns:
            raise KeyError(
                "Missing required HDF5 columns: "
                + ", ".join(missing_columns)
            )

        column_indices = {
            name: column_names.index(name)
            for name in required_columns
        }

        if halo_name.shape[0] != data.shape[0]:
            raise ValueError(
                f"halo_name length {halo_name.shape[0]} does not match "
                f"data length {data.shape[0]}."
            )

        n_total = data.shape[0]

        print()
        print("=" * 80)
        print("Preparing extended and pointlike subhalo components")
        print("=" * 80)
        print(f"Scenario: {scenario}")
        print(f"REPOP_ID: {repop_id}")
        print(f"HDF5 internal iteration: {iteration}")
        print(f"Input HDF5: {input_h5}")
        print(f"Extended CLUMPY list: {output_list}")
        print(f"Pointlike FITS: {output_pointlike_fits}")
        print(f"Total halos in HDF5: {n_total:,}")
        print(f"NSIDE: {nside}")
        print(f"HEALPix ordering: NESTED")
        print(f"Equivalent HEALPix pixel size: {theta_pix_deg:.6f} deg")
        print(f"Extended:  theta_s >= {theta_min_deg:.6f} deg")
        print(f"Pointlike: theta_s <  {theta_min_deg:.6f} deg")
        print(f"TOP_N for extended list: {top_n}")
        print(f"Chunk size: {chunk_size:,}")
        print("=" * 80)
        print()

        with open(output_list, "w", encoding="utf-8") as f:
            write_header(
                f=f,
                input_h5=input_h5,
                output_list=output_list,
                scenario=scenario,
                repop_id=repop_id,
                iteration=iteration,
                nside=nside,
                theta_pix_deg=theta_pix_deg,
                theta_min_deg=theta_min_deg,
                chunk_size=chunk_size,
                halo_type=halo_type,
                top_n=top_n,
            )

            for start_row in range(0, n_total, chunk_size):
                end_row = min(start_row + chunk_size, n_total)

                arr = data[start_row:end_row]
                names = decode_halo_names(
                    halo_name[start_row:end_row]
                )

                total_seen += arr.shape[0]

                js = arr[:, column_indices["Js"]]
                d_earth = arr[:, column_indices["D_Earth"]]
                theta_s = arr[:, column_indices["theta_s"]]
                r_s = arr[:, column_indices["r_s"]]
                rho_s = arr[:, column_indices["rho_s"]]
                x_e = arr[:, column_indices["Xearth"]]
                y_e = arr[:, column_indices["Yearth"]]
                z_e = arr[:, column_indices["Zearth"]]

                radius = np.sqrt(x_e**2 + y_e**2 + z_e**2)

                valid = (
                    np.isfinite(js)
                    & np.isfinite(d_earth)
                    & np.isfinite(theta_s)
                    & np.isfinite(r_s)
                    & np.isfinite(rho_s)
                    & np.isfinite(x_e)
                    & np.isfinite(y_e)
                    & np.isfinite(z_e)
                    & np.isfinite(radius)
                    & (js > 0.0)
                    & (d_earth > 0.0)
                    & (r_s > 0.0)
                    & (rho_s > 0.0)
                    & (radius > 0.0)
                )

                mask_extended = valid & (theta_s >= theta_min_deg)
                mask_pointlike = valid & (theta_s < theta_min_deg)

                n_valid_chunk = int(np.count_nonzero(valid))
                n_extended_chunk = int(np.count_nonzero(mask_extended))
                n_pointlike_chunk = int(np.count_nonzero(mask_pointlike))

                total_valid += n_valid_chunk
                total_invalid += arr.shape[0] - n_valid_chunk
                total_extended += n_extended_chunk
                total_pointlike += n_pointlike_chunk

                if n_pointlike_chunk:
                    js_pointlike = js[mask_pointlike]

                    lon_deg, lat_deg = xyz_to_lb_deg(
                        x_e[mask_pointlike],
                        y_e[mask_pointlike],
                        z_e[mask_pointlike],
                    )

                    pixel = hp.ang2pix(
                        nside,
                        lon_deg,
                        lat_deg,
                        lonlat=True,
                        nest=True,
                    )

                    pointlike_map += np.bincount(
                        pixel,
                        weights=js_pointlike,
                        minlength=pointlike_map.size,
                    )

                    total_pointlike_js += js_pointlike.sum(
                        dtype=np.float64
                    )

                mask_extended_to_write = mask_extended

                if top_n is not None:
                    remaining = top_n - total_extended_written

                    if remaining <= 0:
                        mask_extended_to_write = np.zeros_like(
                            mask_extended,
                            dtype=bool,
                        )
                    else:
                        kept_indices = np.flatnonzero(mask_extended)

                        if kept_indices.size > remaining:
                            limited_mask = np.zeros_like(
                                mask_extended,
                                dtype=bool,
                            )
                            limited_mask[kept_indices[:remaining]] = True
                            mask_extended_to_write = limited_mask

                n_written_chunk = write_rows_for_chunk(
                    f=f,
                    names=names,
                    arr=arr,
                    mask=mask_extended_to_write,
                    halo_type=halo_type,
                    column_indices=column_indices,
                )

                total_extended_written += n_written_chunk

                print(
                    f"Processed rows {start_row:,} - {end_row:,} / "
                    f"{n_total:,} | "
                    f"extended={total_extended:,} | "
                    f"extended written={total_extended_written:,} | "
                    f"pointlike={total_pointlike:,}",
                    flush=True,
                )

    map_sum = pointlike_map.sum(dtype=np.float64)

    if not np.isclose(
        map_sum,
        total_pointlike_js,
        rtol=1e-12,
        atol=0.0,
    ):
        raise RuntimeError(
            "Pointlike J-factor conservation failed: "
            f"sum(map)={map_sum:.16e}, "
            f"sum(catalog)={total_pointlike_js:.16e}"
        )

    write_pointlike_fits(
        output_path=output_pointlike_fits,
        pointlike_map=pointlike_map,
        nside=nside,
        scenario=scenario,
        repop_id=repop_id,
        theta_cut_deg=theta_min_deg,
        n_pointlike=total_pointlike,
    )

    print()
    print("=" * 80)
    print("Finished preparing subhalo components")
    print("=" * 80)
    print(f"Input HDF5: {input_h5}")
    print(f"Extended CLUMPY list: {output_list}")
    print(f"Pointlike FITS: {output_pointlike_fits}")
    print(f"Total halos seen: {total_seen:,}")
    print(f"Valid halos: {total_valid:,}")
    print(f"Invalid halos excluded: {total_invalid:,}")
    print(f"Extended halos: {total_extended:,}")
    print(f"Extended halos written: {total_extended_written:,}")
    print(f"Pointlike halos: {total_pointlike:,}")
    print(f"Pointlike catalog sum(Js): {total_pointlike_js:.16e}")
    print(f"Pointlike map sum:         {map_sum:.16e}")
    print(f"TOP_N for extended list: {top_n}")
    print("=" * 80)

def main():
    args = parse_args()

    repop_id = args.repop_id
    scenario = args.scenario

    if repop_id < 0:
        raise ValueError("repop_id must be a non-negative integer.")

    input_h5 = get_input_h5(repop_id, scenario)
    output_list = get_output_list(repop_id, scenario, TOP_N)
    output_pointlike_fits = get_output_pointlike_fits(
        repop_id,
        scenario,
        NSIDE,
    )

    prepare_subhalo_components(
        input_h5=input_h5,
        output_list=output_list,
        output_pointlike_fits=output_pointlike_fits,
        scenario=scenario,
        repop_id=repop_id,
        iteration=ITERATION,
        top_n=TOP_N,
        halo_type=HALO_TYPE,
        nside=NSIDE,
        round_up_decimals=ROUND_UP_DECIMALS,
        chunk_size=CHUNK_SIZE,
    )


if __name__ == "__main__":
    main()
