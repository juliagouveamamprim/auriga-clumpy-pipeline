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
        repop_0001_raw_nopointlike_nside<NSIDE>.txt

    outputs/clumpy/<scenario>/pointlike/
        repop_0001_pointlike_nside<NSIDE>.fits

    outputs/clumpy/<scenario>/cases/repop_0001_nside<NSIDE>/
        clumpy_params.template.txt
        preparation_manifest.json

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
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
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
TEMPLATE_DIR = REPOSITORY_ROOT / "configs" / "clumpy_templates"
TEMPLATE_NAME = (
    "clumpy_params_g6_auriga_nfw_{scenario}_renorm_vmin0p1.template.txt"
)

# Internal HDF5 iteration index.
# In our one-directory-per-repop convention, each HDF5 contains iteration_0.
ITERATION = 0

# Deprecated: TOP_N is no longer supported because the HDF5 catalogue
# row order is not guaranteed to be sorted by J-factor.
# Keep TOP_N = None and use explicit J-factor cuts instead.
TOP_N = None

# CLUMPY halo type.
HALO_TYPE = "DSPH"

# HEALPix map resolution used to define the point-like cut.
NSIDE = 2048

# Adopted production catalogue cuts. Disabling them requires --no-cuts.
DEFAULT_EXTENDED_CUT_F = 1.0e-3
DEFAULT_POINTLIKE_CUT_F = 1.0e-3

PREPARATION_MANIFEST_SCHEMA_VERSION = 1

SCIENTIFIC_GMW_RHOSOL = {
    "resilient": 3.9447023823e-1,
    "fragile": 3.9570067534e-1,
}

KNOWN_HDF5_GROUP_ATTRIBUTES = (
    "n_generated",
    "n_removed_engulfing",
    "n_removed_roche",
    "n_saved",
)

# Number of decimal places used when rounding theta_pix upward.
# Example: if theta_pix = 0.02863 deg and ROUND_UP_DECIMALS = 2,
# then theta_min_deg = 0.03 deg.
ROUND_UP_DECIMALS = 2

# Number of HDF5 rows read at a time.
#
# 500_000 is conservative for both the legacy 14-column schema
# and the reduced 10-column schema, plus names and temporary arrays.
CHUNK_SIZE = 500_000


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
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

    parser.add_argument(
        "--nside",
        type=int,
        default=NSIDE,
        help=(
            "HEALPix NSIDE used for the pointlike map and for the "
            "automatic pointlike/extended angular threshold."
        ),
    )

    parser.add_argument(
        "--cut-f",
        type=float,
        default=None,
        help=(
            "Set the same cut fraction for extended and pointlike halos. "
            "Cannot be combined with the component-specific cut options."
        ),
    )

    parser.add_argument(
        "--extended-cut-f",
        type=float,
        default=None,
        help=(
            "Cut fraction for extended halos (production default: 1e-3). "
            "extended halos are kept only when their corrected-CLUMPY "
            "central-pixel proxy is >= extended_cut_f * J_pixel_ref."
        ),
    )

    parser.add_argument(
        "--pointlike-cut-f",
        type=float,
        default=None,
        help=(
            "Cut fraction for pointlike halos (production default: 1e-3). "
            "pointlike halos are kept only when Js >= pointlike_cut_f "
            "* J_pixel_ref."
        ),
    )

    parser.add_argument(
        "--no-cuts",
        action="store_true",
        help=(
            "Explicitly disable both extended and pointlike catalogue cuts. "
            "Cannot be combined with either cut-f option."
        ),
    )

    parser.add_argument(
        "--theta-aperture-deg",
        type=float,
        default=None,
        help=(
            "Deprecated. The extended central-pixel proxy now uses the "
            "CLUMPY aperture hp.max_pixrad(NSIDE), derived automatically."
        ),
    )

    args = parser.parse_args(argv)

    if args.no_cuts:
        if (
            args.cut_f is not None
            or args.extended_cut_f is not None
            or args.pointlike_cut_f is not None
        ):
            parser.error(
                "--no-cuts cannot be combined with --cut-f, "
                "--extended-cut-f, or --pointlike-cut-f"
            )
        args.extended_cut_f = None
        args.pointlike_cut_f = None
    else:
        if args.cut_f is not None:
            if (
                args.extended_cut_f is not None
                or args.pointlike_cut_f is not None
            ):
                parser.error(
                    "--cut-f cannot be combined with --extended-cut-f or "
                    "--pointlike-cut-f"
                )
            args.extended_cut_f = args.cut_f
            args.pointlike_cut_f = args.cut_f
        else:
            if args.extended_cut_f is None:
                args.extended_cut_f = DEFAULT_EXTENDED_CUT_F
            if args.pointlike_cut_f is None:
                args.pointlike_cut_f = DEFAULT_POINTLIKE_CUT_F

        for option_name, value in (
            ("--extended-cut-f", args.extended_cut_f),
            ("--pointlike-cut-f", args.pointlike_cut_f),
        ):
            if not math.isfinite(value) or value < 0.0:
                parser.error(f"{option_name} must be finite and non-negative")

    return args


# ============================================================
# Path helpers
# ============================================================

def get_h5_dir(repop_id, base_h5_dir=BASE_H5_DIR):
    """
    Return HDF5 directory for a given global repop ID.

    All repops are expected to follow:

        outputs/repop_XXXX/
    """
    return Path(base_h5_dir) / f"repop_{repop_id:04d}"


def get_input_h5(repop_id, scenario, base_h5_dir=BASE_H5_DIR):
    """
    Return input HDF5 file for one repop/scenario.
    """
    return get_h5_dir(repop_id, base_h5_dir) / f"fullrepop_hydro_{scenario}.h5"


def get_output_list(
    repop_id,
    scenario,
    top_n,
    nside=NSIDE,
    base_run_dir=BASE_RUN_DIR,
):
    """
    Return output CLUMPY raw list path.

    The NSIDE value is always included in the filename to prevent products
    generated at different resolutions from overwriting one another.
    """
    base = f"repop_{repop_id:04d}_raw_nopointlike_nside{nside}"

    if top_n is not None:
        base += f"_top{top_n}"

    filename = base + ".txt"

    return (
        Path(base_run_dir)
        / scenario
        / "lists"
        / "raw"
        / filename
    )


def get_output_pointlike_fits(
    repop_id,
    scenario,
    nside,
    base_run_dir=BASE_RUN_DIR,
):
    """Return output path for the pointlike-only HEALPix FITS map."""
    return (
        Path(base_run_dir)
        / scenario
        / "pointlike"
        / f"repop_{repop_id:04d}_pointlike_nside{nside}.fits"
    )


def get_case_dir(repop_id, scenario, nside, base_run_dir=BASE_RUN_DIR):
    """Return the preparation metadata directory for one exact case."""
    return (
        Path(base_run_dir)
        / scenario
        / "cases"
        / f"repop_{repop_id:04d}_nside{nside}"
    )


def get_template_path(scenario, template_dir=TEMPLATE_DIR):
    """Return the immutable scientific CLUMPY template for a scenario."""
    return Path(template_dir) / TEMPLATE_NAME.format(scenario=scenario)


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 checksum of a file without loading it in memory."""
    digest = hashlib.sha256()

    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def normalize_known_hdf5_group_attributes(attributes):
    """Return validated integer catalogue counters used by preparation."""
    normalized = {}

    for name in KNOWN_HDF5_GROUP_ATTRIBUTES:
        if name not in attributes:
            continue

        value = attributes[name]
        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                raise ValueError(
                    f"HDF5 attribute {name!r} must be a scalar integer."
                )
            value = value.item()
        elif isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"HDF5 attribute {name!r} must be a scalar integer."
            )
        if value < 0:
            raise ValueError(
                f"HDF5 attribute {name!r} must be non-negative."
            )

        normalized[name] = int(value)

    return normalized


def read_gmw_rhosol_from_bytes(template_bytes, source_description="template"):
    """Read the unique finite gMW_RHOSOL value from template bytes."""
    matches = []

    try:
        template_text = template_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"CLUMPY template is not valid UTF-8: {source_description}"
        ) from exc

    for line in template_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0] == "gMW_RHOSOL":
            if len(parts) < 3:
                raise ValueError(
                    "Could not parse gMW_RHOSOL in template: "
                    f"{source_description}"
                )
            if parts[1] != "[GeV/cm3]":
                raise ValueError(
                    "Expected gMW_RHOSOL units [GeV/cm3] in template: "
                    f"{source_description}"
                )
            try:
                value = float(parts[2])
            except ValueError as exc:
                raise ValueError(
                    "Could not parse gMW_RHOSOL in template: "
                    f"{source_description}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    "gMW_RHOSOL must be finite in template: "
                    f"{source_description}"
                )
            matches.append(value)

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one gMW_RHOSOL entry in template "
            f"{source_description}, found {len(matches)}."
        )

    return matches[0]


def read_gmw_rhosol(template_path):
    """Read gMW_RHOSOL from one immutable read of a CLUMPY template."""
    template_path = Path(template_path)
    return read_gmw_rhosol_from_bytes(
        template_path.read_bytes(),
        source_description=str(template_path),
    )


def validate_template_scenario(scenario, gmw_rhosol, source_description):
    """Reject a template whose fixed MW normalization belongs to another case."""
    expected = SCIENTIFIC_GMW_RHOSOL[scenario]
    if not math.isclose(gmw_rhosol, expected, rel_tol=1.0e-12, abs_tol=0.0):
        raise ValueError(
            f"CLUMPY template does not match scenario {scenario!r}: "
            f"gMW_RHOSOL={gmw_rhosol:.10e}, expected {expected:.10e} in "
            f"{source_description}."
        )


def write_bytes_without_overwrite(data, destination):
    """Write bytes while refusing to replace any existing destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "xb") as destination_stream:
        destination_stream.write(data)
        destination_stream.flush()
        os.fsync(destination_stream.fileno())


def validate_finite_scientific_values(value, path="scientific_data"):
    """Reject non-finite numbers recursively before manifest publication."""
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite_scientific_values(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_finite_scientific_values(item, f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite scientific value at {path}: {value!r}")


def write_json_atomic_no_overwrite(path, payload):
    """Publish a complete JSON file atomically without overwriting a target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)

        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        # link() fails if the destination appeared after the preflight check.
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def raise_for_existing_targets(targets):
    """Refuse preparation if any case product already exists."""
    existing = [Path(path) for path in targets if os.path.lexists(path)]

    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Preparation refused because target product(s) already exist:\n"
            f"{formatted}\n"
            "No files were changed. Choose another case or inspect the "
            "existing products. --force is not implemented yet."
        )


class PreparationCaseLock:
    """Exclusive per-case lock removed on both success and failure."""

    def __init__(self, path, repop_id, scenario, nside):
        self.path = Path(path)
        self.repop_id = repop_id
        self.scenario = scenario
        self.nside = nside
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                "Preparation refused because this exact case is already "
                f"locked: {self.path}"
            ) from exc

        self.acquired = True
        try:
            payload = (
                f"pid={os.getpid()}\n"
                f"repop_id={self.repop_id}\n"
                f"scenario={self.scenario}\n"
                f"nside={self.nside}\n"
            ).encode("utf-8")
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            if self.acquired:
                self.path.unlink(missing_ok=True)
                self.acquired = False
            raise

        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False
        return False


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
    cut_metadata=None,
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

    if cut_metadata is None:
        header.extend(
            [
                "# Subhalo cuts: disabled",
                "#",
            ]
        )
    else:
        header.extend(
            [
                "# Subhalo cuts: enabled",
                f"# theta_aperture_deg = {cut_metadata['theta_aperture_deg']:.8e}",
                f"# brightest_pointlike_pixel_theory = {cut_metadata['brightest_pointlike']:.8e}",
                f"# brightest_extended_pixel_theory = {cut_metadata['brightest_extended']:.8e}",
                f"# J_pixel_ref = {cut_metadata['j_pixel_ref']:.8e}",
                f"# extended_cut_f = {cut_metadata['extended_cut_f']}",
                f"# pointlike_cut_f = {cut_metadata['pointlike_cut_f']}",
                f"# extended_J_cut = {cut_metadata['extended_j_cut']}",
                f"# pointlike_J_cut = {cut_metadata['pointlike_j_cut']}",
                "# Extended cut: keep if central_pixel_proxy >= extended_J_cut",
                "# Pointlike cut: keep if Js >= pointlike_J_cut",
                "#",
            ]
        )

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
    n_pointlike_total=None,
    cut_metadata=None,
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
        hdu.header["OBJECT"] = "PARTIAL"
        hdu.header["SCENARIO"] = scenario
        hdu.header["REPOPID"] = repop_id
        hdu.header["THETACUT"] = (
            theta_cut_deg,
            "Pointlike cut: theta_s < THETACUT [deg]",
        )
        hdu.header["NPOINT"] = (
            n_pointlike,
            "Number of pointlike subhalos in this map",
        )
        if n_pointlike_total is not None:
            hdu.header["NPTOTAL"] = (
                n_pointlike_total,
                "Number of pointlike subhalos before cuts",
            )
        if cut_metadata is None:
            hdu.header["CUTS"] = (False, "Subhalo cuts applied")
        else:
            hdu.header["CUTS"] = (True, "Subhalo cuts applied")
            hdu.header["THETAAP"] = (
                float(cut_metadata["theta_aperture_deg"]),
                "CLUMPY max-pixel-radius aperture [deg]",
            )
            hdu.header["JREF"] = (
                float(cut_metadata["j_pixel_ref"]),
                "Brightest theoretical pixel reference",
            )
            if cut_metadata["pointlike_cut_f"] is not None:
                hdu.header["FPL"] = (
                    float(cut_metadata["pointlike_cut_f"]),
                    "Pointlike cut fraction",
                )
                hdu.header["JCPL"] = (
                    float(cut_metadata["pointlike_j_cut"]),
                    "Pointlike J cut",
                )
            if cut_metadata["extended_cut_f"] is not None:
                hdu.header["FEXT"] = (
                    float(cut_metadata["extended_cut_f"]),
                    "Extended cut fraction",
                )
                hdu.header["JCEXT"] = (
                    float(cut_metadata["extended_j_cut"]),
                    "Extended central-pixel proxy cut",
                )
            hdu.header["BPL"] = (
                float(cut_metadata["brightest_pointlike"]),
                "Brightest pointlike proxy",
            )
            hdu.header["BEXT"] = (
                float(cut_metadata["brightest_extended"]),
                "Brightest extended proxy",
            )
        hdu.header["PIXAREA"] = (
            pixel_area_sr,
            "HEALPix pixel solid angle [sr]",
        )

    primary = fits.PrimaryHDU()
    primary.header["CONTENT"] = "Auriga pointlike subhalo J-factor map"

    hdul = fits.HDUList(
        [
            primary,
            hdu_j,
            hdu_per_sr,
        ]
    )
    descriptor = os.open(
        output_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "wb") as output_stream:
        hdul.writeto(output_stream)



def projected_nfw_fraction(y):
    """
    Return the projected annihilation-J fraction for a truncated NFW halo.

    The halo is truncated at r_s, consistently with the CLUMPY lists
    produced by this pipeline. The dimensionless projected aperture is

        y = D_Earth * sin(alpha_int) / r_s.

    The returned fraction is normalized to the HDF5 Js, which is the
    annihilation J-factor integrated over the halo up to r_s.

    This is the closed-form result for a circular aperture centred on
    the halo. Series expansions are used near y = 0 and y = 1 to avoid
    catastrophic cancellation.
    """

    y = np.asarray(y, dtype=np.float64)
    fraction = np.zeros_like(y)

    fraction[y >= 1.0] = 1.0

    valid = (y > 0.0) & (y < 1.0)
    small = valid & (y < 1.0e-4)

    if np.any(small):
        z = y[small]

        fraction[small] = (
            12.0 * np.pi * z / 7.0
            + z**2
            * (
                48.0 * np.log(z) / 7.0
                - 11.0 / 14.0
            )
            + z**4
            * (
                60.0 * np.log(z) / 7.0
                + 83.0 / 56.0
            )
        )

    near_one = (
        valid
        & ~small
        & ((1.0 - y) < 1.0e-4)
    )

    if np.any(near_one):
        eps = 1.0 - y[near_one]

        fraction[near_one] = (
            1.0
            - np.sqrt(2.0) * eps**1.5 / 7.0
            - 17.0 * np.sqrt(2.0) * eps**2.5 / 140.0
            - 809.0 * np.sqrt(2.0) * eps**3.5 / 7840.0
        )

    regular = valid & ~small & ~near_one

    if np.any(regular):
        z = y[regular]

        numerator = (
            7.0
            - 36.0 * z**2
            + 45.0 * z**4
            - 16.0 * z**6
            - 12.0
            * z**2
            * (2.0 * z**4 - 5.0 * z**2 + 4.0)
            * np.log(z)
        )

        excluded_fraction = (
            -z * np.arccos(z)
            + numerator
            / (
                24.0
                * (1.0 - z**2) ** 2.5
            )
        )

        fraction[regular] = (
            1.0
            - 24.0 * excluded_fraction / 7.0
        )

    fraction[~np.isfinite(fraction)] = 0.0

    return np.clip(fraction, 0.0, 1.0)


def clumpy_central_pixel_proxy_from_js(
    js,
    d_earth_kpc,
    rs_kpc,
    nside,
):
    """
    Estimate the corrected-CLUMPY central-pixel contribution.

    CLUMPY evaluates the J-factor in a circular aperture whose radius is

        alpha_int = hp.max_pixrad(nside),

    and then rescales the aperture-integrated value by

        Omega_pixel / Omega_aperture.

    The projected NFW fraction is evaluated analytically, so this function
    requires no numerical integration per subhalo.
    """

    alpha_int_rad = float(hp.max_pixrad(nside))
    omega_pixel = float(hp.nside2pixarea(nside))
    omega_aperture = float(
        2.0
        * np.pi
        * (1.0 - np.cos(alpha_int_rad))
    )

    js = np.asarray(js, dtype=np.float64)
    d_earth_kpc = np.asarray(d_earth_kpc, dtype=np.float64)
    rs_kpc = np.asarray(rs_kpc, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y = (
            d_earth_kpc
            * np.sin(alpha_int_rad)
            / rs_kpc
        )

        projected_fraction = projected_nfw_fraction(y)

        j_pixel_proxy = (
            js
            * projected_fraction
            * omega_pixel
            / omega_aperture
        )

    j_pixel_proxy = np.asarray(j_pixel_proxy, dtype=np.float64)
    j_pixel_proxy[~np.isfinite(j_pixel_proxy)] = 0.0
    j_pixel_proxy[j_pixel_proxy < 0.0] = 0.0

    return j_pixel_proxy

def build_valid_mask(js, d_earth, theta_s, r_s, rho_s, x_e, y_e, z_e):
    """Return the validity mask used consistently in both HDF5 passes."""

    radius = np.sqrt(x_e**2 + y_e**2 + z_e**2)

    return (
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


def compute_pixel_reference(
    data,
    column_indices,
    n_total,
    theta_min_deg,
    nside,
    chunk_size,
    progress_label=None,
):
    """
    Compute the diagnostic-inspired brightest theoretical pixel reference.

    Pointlike halos enter with J_pixel_proxy = Js.
    Extended halos enter with the corrected-CLUMPY central-pixel proxy.
    """

    brightest_pointlike = 0.0
    brightest_extended = 0.0

    for start_row in range(0, n_total, chunk_size):
        end_row = min(start_row + chunk_size, n_total)
        arr = data[start_row:end_row]

        js = arr[:, column_indices["Js"]]
        d_earth = arr[:, column_indices["D_Earth"]]
        theta_s = arr[:, column_indices["theta_s"]]
        r_s = arr[:, column_indices["r_s"]]
        rho_s = arr[:, column_indices["rho_s"]]
        x_e = arr[:, column_indices["Xearth"]]
        y_e = arr[:, column_indices["Yearth"]]
        z_e = arr[:, column_indices["Zearth"]]

        valid = build_valid_mask(
            js=js,
            d_earth=d_earth,
            theta_s=theta_s,
            r_s=r_s,
            rho_s=rho_s,
            x_e=x_e,
            y_e=y_e,
            z_e=z_e,
        )

        mask_pointlike = valid & (theta_s < theta_min_deg)
        mask_extended = valid & (theta_s >= theta_min_deg)

        if np.any(mask_pointlike):
            local_max = float(np.nanmax(js[mask_pointlike]))
            brightest_pointlike = max(brightest_pointlike, local_max)

        if np.any(mask_extended):
            ext_pixel_proxy = clumpy_central_pixel_proxy_from_js(
                js=js[mask_extended],
                d_earth_kpc=d_earth[mask_extended],
                rs_kpc=r_s[mask_extended],
                nside=nside,
            )
            local_max = float(np.nanmax(ext_pixel_proxy))
            brightest_extended = max(brightest_extended, local_max)

        if progress_label is not None:
            print(
                f"{progress_label}: {end_row:,} / {n_total:,} rows "
                f"({100.0 * end_row / n_total:.1f}%)",
                flush=True,
            )

    j_pixel_ref = max(brightest_pointlike, brightest_extended)

    if not np.isfinite(j_pixel_ref) or j_pixel_ref <= 0.0:
        raise RuntimeError(
            "Could not compute a positive J_pixel_ref for subhalo cuts."
        )

    return {
        "brightest_pointlike": brightest_pointlike,
        "brightest_extended": brightest_extended,
        "j_pixel_ref": j_pixel_ref,
    }


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
    extended_cut_f=DEFAULT_EXTENDED_CUT_F,
    pointlike_cut_f=DEFAULT_POINTLIKE_CUT_F,
    theta_aperture_deg=None,
):
    """
    Build the extended CLUMPY list and pointlike HEALPix map.

    Without cuts, this uses a single chunked pass through the HDF5 catalog.
    With cuts enabled, it first computes the global theoretical brightest
    pixel proxy and then performs a second pass to write only kept halos.
    """

    if top_n is not None:
        raise ValueError(
            "top_n is deprecated because the HDF5 catalogue row order is "
            "no longer guaranteed to be sorted by J-factor. Use explicit "
            "J-factor cuts instead."
        )

    input_h5 = Path(input_h5)
    output_list = Path(output_list)
    output_pointlike_fits = Path(output_pointlike_fits)

    raise_for_existing_targets([output_list, output_pointlike_fits])

    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_h5}")

    theta_pix_deg = healpix_pixel_size_deg(nside)
    theta_min_deg = round_up(theta_pix_deg, decimals=round_up_decimals)

    cuts_enabled = (
        extended_cut_f is not None
        or pointlike_cut_f is not None
    )

    for cut_name, cut_value in (
        ("extended_cut_f", extended_cut_f),
        ("pointlike_cut_f", pointlike_cut_f),
    ):
        if cut_value is None:
            continue
        if not math.isfinite(cut_value) or cut_value < 0.0:
            raise ValueError(f"{cut_name} must be finite and non-negative.")

    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_pointlike_fits.parent.mkdir(parents=True, exist_ok=True)

    alpha_int_rad = float(hp.max_pixrad(nside))
    alpha_int_deg = float(np.rad2deg(alpha_int_rad))

    if theta_aperture_deg is not None:
        raise ValueError(
            "--theta-aperture-deg is deprecated. The aperture is now "
            "derived automatically as hp.max_pixrad(NSIDE)."
        )

    # Keep this internal name temporarily for backward-compatible metadata.
    theta_aperture_deg = alpha_int_deg

    group_name = f"iteration_{iteration}"

    pointlike_map = np.zeros(
        hp.nside2npix(nside),
        dtype=np.float64,
    )

    total_seen = 0
    total_valid = 0
    total_invalid = 0
    valid_js_subtotals = []

    total_extended = 0
    total_extended_cut_kept = 0
    total_extended_written = 0
    extended_js_subtotals = []
    extended_kept_js_subtotals = []
    extended_discarded_js_subtotals = []

    total_pointlike = 0
    total_pointlike_kept = 0
    pointlike_js_subtotals = []
    pointlike_kept_js_subtotals = []
    pointlike_discarded_js_subtotals = []

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

        group_attributes = normalize_known_hdf5_group_attributes(group.attrs)

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
        data_attributes = {"column_names": column_names}

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

        cut_metadata = None
        extended_j_cut = None
        pointlike_j_cut = None

        if cuts_enabled:
            print("Subhalo cuts: enabled")
            print(
                "CLUMPY integration aperture: "
                f"{theta_aperture_deg:.12f} deg"
            )
            print(f"Extended cut fraction: {extended_cut_f}")
            print(f"Pointlike cut fraction: {pointlike_cut_f}")
            print("Computing theoretical brightest pixel reference...")

            ref_info = compute_pixel_reference(
                data=data,
                column_indices=column_indices,
                n_total=n_total,
                theta_min_deg=theta_min_deg,
                nside=nside,
                chunk_size=chunk_size,
            )

            j_pixel_ref = ref_info["j_pixel_ref"]

            if extended_cut_f is not None:
                extended_j_cut = extended_cut_f * j_pixel_ref

            if pointlike_cut_f is not None:
                pointlike_j_cut = pointlike_cut_f * j_pixel_ref

            cut_metadata = {
                "theta_aperture_deg": theta_aperture_deg,
                "brightest_pointlike": ref_info["brightest_pointlike"],
                "brightest_extended": ref_info["brightest_extended"],
                "j_pixel_ref": j_pixel_ref,
                "extended_cut_f": extended_cut_f,
                "pointlike_cut_f": pointlike_cut_f,
                "extended_j_cut": extended_j_cut,
                "pointlike_j_cut": pointlike_j_cut,
            }
            validate_finite_scientific_values(cut_metadata, "cuts")

            print(f"brightest pointlike proxy: {ref_info['brightest_pointlike']:.8e}")
            print(f"brightest extended proxy:  {ref_info['brightest_extended']:.8e}")
            print(f"J_pixel_ref:               {j_pixel_ref:.8e}")
            print(f"Extended J cut:            {extended_j_cut}")
            print(f"Pointlike J cut:           {pointlike_j_cut}")
        else:
            print("Subhalo cuts: disabled")

        print("=" * 80)
        print()

        with open(output_list, "x", encoding="utf-8") as f:
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
                cut_metadata=cut_metadata,
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

                valid = build_valid_mask(
                    js=js,
                    d_earth=d_earth,
                    theta_s=theta_s,
                    r_s=r_s,
                    rho_s=rho_s,
                    x_e=x_e,
                    y_e=y_e,
                    z_e=z_e,
                )

                mask_extended = valid & (theta_s >= theta_min_deg)
                mask_pointlike = valid & (theta_s < theta_min_deg)

                mask_extended_kept = mask_extended
                mask_pointlike_kept = mask_pointlike

                if extended_j_cut is not None and np.any(mask_extended):
                    ext_pixel_proxy = (
                        clumpy_central_pixel_proxy_from_js(
                            js=js[mask_extended],
                            d_earth_kpc=d_earth[mask_extended],
                            rs_kpc=r_s[mask_extended],
                            nside=nside,
                        )
                    )
                    local_ext_keep = (
                        ext_pixel_proxy >= extended_j_cut
                    )
                    mask_extended_kept = np.zeros_like(mask_extended)
                    mask_extended_kept[np.flatnonzero(mask_extended)] = (
                        local_ext_keep
                    )

                if pointlike_j_cut is not None:
                    mask_pointlike_kept = mask_pointlike & (js >= pointlike_j_cut)

                mask_extended_discarded = mask_extended & ~mask_extended_kept
                mask_pointlike_discarded = mask_pointlike & ~mask_pointlike_kept

                n_valid_chunk = int(np.count_nonzero(valid))
                n_extended_chunk = int(np.count_nonzero(mask_extended))
                n_extended_kept_chunk = int(np.count_nonzero(mask_extended_kept))
                n_pointlike_chunk = int(np.count_nonzero(mask_pointlike))
                n_pointlike_kept_chunk = int(np.count_nonzero(mask_pointlike_kept))

                total_valid += n_valid_chunk
                total_invalid += arr.shape[0] - n_valid_chunk
                total_extended += n_extended_chunk
                total_extended_cut_kept += n_extended_kept_chunk
                total_pointlike += n_pointlike_chunk
                total_pointlike_kept += n_pointlike_kept_chunk

                if n_valid_chunk:
                    valid_js_subtotals.append(
                        float(js[valid].sum(dtype=np.float64))
                    )

                if n_extended_chunk:
                    extended_js_subtotals.append(
                        float(js[mask_extended].sum(dtype=np.float64))
                    )

                if n_extended_kept_chunk:
                    extended_kept_js_subtotals.append(
                        float(js[mask_extended_kept].sum(dtype=np.float64))
                    )

                if np.any(mask_extended_discarded):
                    extended_discarded_js_subtotals.append(
                        float(js[mask_extended_discarded].sum(dtype=np.float64))
                    )

                if n_pointlike_chunk:
                    pointlike_js_subtotals.append(
                        float(js[mask_pointlike].sum(dtype=np.float64))
                    )

                if n_pointlike_kept_chunk:
                    js_pointlike = js[mask_pointlike_kept]

                    lon_deg, lat_deg = xyz_to_lb_deg(
                        x_e[mask_pointlike_kept],
                        y_e[mask_pointlike_kept],
                        z_e[mask_pointlike_kept],
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

                    pointlike_kept_js_subtotals.append(
                        float(js_pointlike.sum(dtype=np.float64))
                    )

                if np.any(mask_pointlike_discarded):
                    pointlike_discarded_js_subtotals.append(
                        float(js[mask_pointlike_discarded].sum(dtype=np.float64))
                    )

                mask_extended_to_write = mask_extended_kept

                if top_n is not None:
                    remaining = top_n - total_extended_written

                    if remaining <= 0:
                        mask_extended_to_write = np.zeros_like(
                            mask_extended,
                            dtype=bool,
                        )
                    else:
                        kept_indices = np.flatnonzero(mask_extended_to_write)

                        if kept_indices.size > remaining:
                            limited_mask = np.zeros_like(
                                mask_extended_to_write,
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
                    f"extended kept={total_extended_cut_kept:,} | "
                    f"extended written={total_extended_written:,} | "
                    f"pointlike={total_pointlike:,} | "
                    f"pointlike kept={total_pointlike_kept:,}",
                    flush=True,
                )

    try:
        total_valid_js = math.fsum(valid_js_subtotals)
        total_extended_js = math.fsum(extended_js_subtotals)
        total_extended_kept_js = math.fsum(extended_kept_js_subtotals)
        total_extended_discarded_js = math.fsum(
            extended_discarded_js_subtotals
        )
        total_pointlike_js = math.fsum(pointlike_js_subtotals)
        total_pointlike_kept_js = math.fsum(pointlike_kept_js_subtotals)
        total_pointlike_discarded_js = math.fsum(
            pointlike_discarded_js_subtotals
        )
    except OverflowError as exc:
        raise ValueError("Scientific J-factor statistics overflowed.") from exc

    map_sum = float(pointlike_map.sum(dtype=np.float64))

    validate_finite_scientific_values(
        {
            "valid_js_sum": total_valid_js,
            "extended_before_cuts_js_sum": total_extended_js,
            "extended_after_cuts_js_sum": total_extended_kept_js,
            "extended_discarded_js_sum": total_extended_discarded_js,
            "pointlike_before_cuts_js_sum": total_pointlike_js,
            "pointlike_after_cuts_js_sum": total_pointlike_kept_js,
            "pointlike_discarded_js_sum": total_pointlike_discarded_js,
            "pointlike_map_js_sum": map_sum,
        },
        "catalogue_statistics",
    )

    if not np.isclose(
        map_sum,
        total_pointlike_kept_js,
        rtol=1e-12,
        atol=0.0,
    ):
        raise RuntimeError(
            "Pointlike J-factor conservation failed: "
            f"sum(map)={map_sum:.16e}, "
            f"sum(catalog kept)={total_pointlike_kept_js:.16e}"
        )

    write_pointlike_fits(
        output_path=output_pointlike_fits,
        pointlike_map=pointlike_map,
        nside=nside,
        scenario=scenario,
        repop_id=repop_id,
        theta_cut_deg=theta_min_deg,
        n_pointlike=total_pointlike_kept,
        n_pointlike_total=total_pointlike,
        cut_metadata=cut_metadata,
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
    print(f"Valid catalog sum(Js): {total_valid_js:.16e}")
    print(f"Extended halos: {total_extended:,}")
    print(f"Extended halos kept by cut: {total_extended_cut_kept:,}")
    print(f"Extended halos written: {total_extended_written:,}")
    print(f"Extended catalog sum(Js), all:  {total_extended_js:.16e}")
    print(f"Extended catalog sum(Js), kept: {total_extended_kept_js:.16e}")
    print(f"Pointlike halos: {total_pointlike:,}")
    print(f"Pointlike halos kept by cut: {total_pointlike_kept:,}")
    print(f"Pointlike catalog sum(Js), all:  {total_pointlike_js:.16e}")
    print(f"Pointlike catalog sum(Js), kept: {total_pointlike_kept_js:.16e}")
    print(f"Pointlike map sum:              {map_sum:.16e}")
    print(f"TOP_N for extended list: {top_n}")
    print("=" * 80)

    cuts = {
        "enabled": cuts_enabled,
        "extended_cut_f": extended_cut_f,
        "pointlike_cut_f": pointlike_cut_f,
        "j_pixel_ref": None,
        "extended_j_cut": None,
        "pointlike_j_cut": None,
        "brightest_pointlike": None,
        "brightest_extended": None,
    }

    if cut_metadata is not None:
        cuts.update({
            "j_pixel_ref": cut_metadata["j_pixel_ref"],
            "extended_j_cut": cut_metadata["extended_j_cut"],
            "pointlike_j_cut": cut_metadata["pointlike_j_cut"],
            "brightest_pointlike": cut_metadata["brightest_pointlike"],
            "brightest_extended": cut_metadata["brightest_extended"],
        })

    return {
        "hdf5": {
            "group": group_name,
            "rows": n_total,
            "group_attributes": group_attributes,
            "data_attributes": data_attributes,
        },
        "geometry": {
            "nside": nside,
            "ordering": "NESTED",
            "theta_pixel_deg": theta_pix_deg,
            "theta_min_deg": theta_min_deg,
            "clumpy_aperture_deg": theta_aperture_deg,
        },
        "cuts": cuts,
        "catalogue": {
            "rows_seen": total_seen,
            "valid_count": total_valid,
            "invalid_count": total_invalid,
            # Invalid rows need not have a finite, physical Js to sum.
            "valid_js_sum": total_valid_js,
            "extended": {
                "before_cuts_count": total_extended,
                "after_cuts_count": total_extended_cut_kept,
                "discarded_count": (
                    total_extended - total_extended_cut_kept
                ),
                "written_count": total_extended_written,
                "before_cuts_js_sum": total_extended_js,
                "after_cuts_js_sum": total_extended_kept_js,
                "discarded_js_sum": total_extended_discarded_js,
            },
            "pointlike": {
                "before_cuts_count": total_pointlike,
                "after_cuts_count": total_pointlike_kept,
                "discarded_count": total_pointlike - total_pointlike_kept,
                "before_cuts_js_sum": total_pointlike_js,
                "after_cuts_js_sum": total_pointlike_kept_js,
                "discarded_js_sum": total_pointlike_discarded_js,
                "map_js_sum": map_sum,
            },
        },
    }


def prepare_case(
    repop_id,
    scenario,
    nside=NSIDE,
    extended_cut_f=DEFAULT_EXTENDED_CUT_F,
    pointlike_cut_f=DEFAULT_POINTLIKE_CUT_F,
    base_h5_dir=BASE_H5_DIR,
    base_run_dir=BASE_RUN_DIR,
    template_path=None,
    repository_root=REPOSITORY_ROOT,
    iteration=ITERATION,
    halo_type=HALO_TYPE,
    round_up_decimals=ROUND_UP_DECIMALS,
    chunk_size=CHUNK_SIZE,
    theta_aperture_deg=None,
):
    """Prepare one case and publish its immutable preparation manifest."""
    if repop_id < 0:
        raise ValueError("repop_id must be a non-negative integer.")
    if scenario not in ("resilient", "fragile"):
        raise ValueError("scenario must be 'resilient' or 'fragile'.")
    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of two.")
    for cut_name, cut_value in (
        ("extended_cut_f", extended_cut_f),
        ("pointlike_cut_f", pointlike_cut_f),
    ):
        if cut_value is None:
            continue
        if not math.isfinite(cut_value) or cut_value < 0.0:
            raise ValueError(f"{cut_name} must be finite and non-negative.")

    # Kept in the signature for compatibility; manifest paths no longer use it.
    del repository_root
    base_run_dir = Path(base_run_dir).resolve()
    input_h5 = get_input_h5(repop_id, scenario, base_h5_dir)
    output_list = get_output_list(
        repop_id,
        scenario,
        TOP_N,
        nside=nside,
        base_run_dir=base_run_dir,
    )
    output_pointlike_fits = get_output_pointlike_fits(
        repop_id,
        scenario,
        nside,
        base_run_dir=base_run_dir,
    )
    case_dir = get_case_dir(
        repop_id,
        scenario,
        nside,
        base_run_dir=base_run_dir,
    )
    template_snapshot = case_dir / "clumpy_params.template.txt"
    manifest_path = case_dir / "preparation_manifest.json"
    lock_path = case_dir / ".preparation.lock"

    if template_path is None:
        template_path = get_template_path(scenario)
    template_path = Path(template_path)

    targets = [
        output_list,
        output_pointlike_fits,
        template_snapshot,
        manifest_path,
    ]
    raise_for_existing_targets(targets)

    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_h5}")
    if not template_path.exists():
        raise FileNotFoundError(f"CLUMPY template not found: {template_path}")

    template_bytes = template_path.read_bytes()
    gmw_rhosol = read_gmw_rhosol_from_bytes(
        template_bytes,
        source_description=str(template_path),
    )
    validate_template_scenario(scenario, gmw_rhosol, str(template_path))

    with PreparationCaseLock(
        lock_path,
        repop_id=repop_id,
        scenario=scenario,
        nside=nside,
    ):
        # Close the preflight/lock race before producing the first artifact.
        raise_for_existing_targets(targets)

        statistics = prepare_subhalo_components(
            input_h5=input_h5,
            output_list=output_list,
            output_pointlike_fits=output_pointlike_fits,
            scenario=scenario,
            repop_id=repop_id,
            iteration=iteration,
            top_n=TOP_N,
            halo_type=halo_type,
            nside=nside,
            round_up_decimals=round_up_decimals,
            chunk_size=chunk_size,
            extended_cut_f=extended_cut_f,
            pointlike_cut_f=pointlike_cut_f,
            theta_aperture_deg=theta_aperture_deg,
        )

        write_bytes_without_overwrite(template_bytes, template_snapshot)

        manifest_base = manifest_path.parent.resolve()

        def manifest_relative(path):
            return Path(
                os.path.relpath(Path(path).resolve(), manifest_base)
            ).as_posix()

        scientific_configuration = {
            "halo_type": halo_type,
            "profile": "kZHAO",
            "profile_parameters": [1, 3, 1],
            "rdelta": "r_s",
            "clumpy_rhos": "rho_s / 4",
            "nside": nside,
            "healpix_ordering": "NESTED",
            "theta_pixel_deg": statistics["geometry"]["theta_pixel_deg"],
            "theta_min_deg": statistics["geometry"]["theta_min_deg"],
            "clumpy_aperture_deg": statistics["geometry"][
                "clumpy_aperture_deg"
            ],
            "cuts": statistics["cuts"],
            "smooth_milky_way": {
                "gMW_RHOSOL_GeV_cm3": gmw_rhosol,
                "normalization_mode": "fixed_by_scenario_template",
            },
        }
        catalogue_statistics = {
            "hdf5": statistics["hdf5"],
            "preparation": statistics["catalogue"],
        }
        validate_finite_scientific_values(
            scientific_configuration,
            "scientific_configuration",
        )
        validate_finite_scientific_values(
            catalogue_statistics,
            "catalogue_statistics",
        )

        manifest = {
            "schema_version": PREPARATION_MANIFEST_SCHEMA_VERSION,
            "state": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": {
                "repop_id": repop_id,
                "repop_tag": f"repop_{repop_id:04d}",
                "scenario": scenario,
                "nside": nside,
                "hdf5_iteration": iteration,
            },
            "paths": {
                "base": ".",
                "base_description": "directory containing this manifest",
            },
            "source": {
                "hdf5": {
                    "path": manifest_relative(input_h5),
                    "provenance_only": True,
                    "required_by_rendering": False,
                },
                "template": {
                    "path": manifest_relative(template_path),
                    "provenance_only": True,
                    "required_by_rendering": False,
                },
            },
            "scientific_configuration": scientific_configuration,
            "catalogue_statistics": catalogue_statistics,
            "artifacts": {
                "raw_list": {
                    "path": manifest_relative(output_list),
                    "sha256": sha256_file(output_list),
                    "size_bytes": output_list.stat().st_size,
                },
                "pointlike_fits": {
                    "path": manifest_relative(output_pointlike_fits),
                    "sha256": sha256_file(output_pointlike_fits),
                    "size_bytes": output_pointlike_fits.stat().st_size,
                },
                "template_snapshot": {
                    "path": manifest_relative(template_snapshot),
                    "sha256": hashlib.sha256(template_bytes).hexdigest(),
                    "size_bytes": template_snapshot.stat().st_size,
                },
            },
        }

        write_json_atomic_no_overwrite(manifest_path, manifest)

    print()
    print("=" * 80)
    print("Preparation case completed")
    print("=" * 80)
    print(f"Template snapshot: {template_snapshot}")
    print(f"Preparation manifest: {manifest_path}")
    print(f"gMW_RHOSOL: {gmw_rhosol:.10e} GeV/cm^3")
    print("CLUMPY was not executed.")
    print("=" * 80)

    return manifest_path, manifest

def main():
    args = parse_args()

    repop_id = args.repop_id
    scenario = args.scenario

    if repop_id < 0:
        raise ValueError("repop_id must be a non-negative integer.")

    nside = args.nside

    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of two.")

    prepare_case(
        repop_id=repop_id,
        scenario=scenario,
        nside=nside,
        extended_cut_f=args.extended_cut_f,
        pointlike_cut_f=args.pointlike_cut_f,
        theta_aperture_deg=args.theta_aperture_deg,
    )


if __name__ == "__main__":
    main()
