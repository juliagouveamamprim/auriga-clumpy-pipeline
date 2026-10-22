#!/usr/bin/env python3
"""Compute the intrinsic substructure boost outside a Galactic latitude mask.

For every saved subhalo, this complementary product uses the original HDF5
NFW parameters and keeps the whole intrinsic luminosity only when the halo
centre satisfies ``|b| >= b_cut``.  It does not use CLUMPY maps, rendering
corrections, catalogue cuts, or pointlike/extended classifications.

The smooth-host denominator is the original, unrenormalized NFW luminosity in
the same unmasked directions.  It is evaluated as a volume integral with a
deterministic adaptive quadrature in Galactocentric spherical coordinates.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from astropy import units as u
from scipy.integrate import quad


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_repopulation_boost import (  # noqa: E402
    ACCUMULATION_DTYPE,
    DATASET_NAME,
    ITERATION_GROUP,
    REPOSITORY_ROOT,
    SUBHALO_LUMINOSITY_COEFFICIENT,
    HostParameters,
    _format_extended,
    catalogue_path,
    host_luminosity,
    read_host_parameters,
    validate_host_consistency,
)


DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_START_ID = 0
DEFAULT_STOP_ID = 500
DEFAULT_MASK_B_CUT_DEG = 10.0
DEFAULT_SMOOTH_RTOL = 1.0e-8
VALID_SCENARIOS = ("fragile", "resilient")

REQUIRED_COLUMNS = ("D_Earth", "Zearth", "r_s", "rho_s")
EXPECTED_COLUMN_UNITS = {
    "D_Earth": u.kpc,
    "Zearth": u.kpc,
    "r_s": u.kpc,
    "rho_s": u.Msun / u.kpc**3,
}

CSV_FIELDS = (
    "repop_id",
    "boost_factor_masked",
    "boost_factor_global",
    "L_sub_unmasked",
    "L_sub_total",
    "L_MW_unmasked",
    "L_MW_total",
    "n_subhalos_unmasked",
    "n_subhalos_total",
    "mask_b_cut_deg",
)


@dataclass(frozen=True)
class MaskedBoostResult:
    """One row of the masked intrinsic-boost product."""

    repop_id: int
    boost_factor_masked: np.longdouble
    boost_factor_global: np.longdouble
    l_sub_unmasked: np.longdouble
    l_sub_total: np.longdouble
    l_mw_unmasked: np.longdouble
    l_mw_total: np.longdouble
    n_subhalos_unmasked: int
    n_subhalos_total: int
    mask_b_cut_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing repop_XXXX catalogue directories.",
    )
    parser.add_argument(
        "--scenario",
        choices=VALID_SCENARIOS,
        required=True,
        help="Disruption scenario to scan.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=DEFAULT_START_ID,
        help="First repopulation ID (inclusive; default: 0).",
    )
    parser.add_argument(
        "--stop-id",
        type=int,
        default=DEFAULT_STOP_ID,
        help="Final repopulation ID (exclusive; default: 500).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of HDF5 rows read per chunk (default: 1000000).",
    )
    parser.add_argument(
        "--mask-b-cut-deg",
        type=float,
        default=DEFAULT_MASK_B_CUT_DEG,
        help="Exclude |b| below this latitude in degrees (default: 10).",
    )
    parser.add_argument(
        "--smooth-rtol",
        type=float,
        default=DEFAULT_SMOOTH_RTOL,
        help="Relative tolerance for the smooth-halo quadrature (default: 1e-8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV. By default, writes outputs/repopulation_boost/"
            "substructure_boost_masked_<scenario>_0500.csv."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve valid existing rows and compute only missing IDs.",
    )
    return parser.parse_args()


def default_output_path(scenario: str) -> Path:
    return (
        REPOSITORY_ROOT
        / "outputs"
        / "repopulation_boost"
        / f"substructure_boost_masked_{scenario}_0500.csv"
    )


def _decode_string_array(values: Iterable[object]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _read_text_scalar(handle: h5py.File, location: str, path: Path) -> str:
    if location not in handle:
        raise KeyError(f"Missing required dataset {location!r} in {path}")
    dataset = handle[location]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Expected a dataset at {location!r} in {path}")
    raw = np.asarray(dataset[...])
    if raw.shape != ():
        raise ValueError(
            f"Expected one string at {location!r} in {path}; found shape {raw.shape}."
        )
    value = raw.item()
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def read_observer_position(handle: h5py.File, path: Path) -> np.ndarray:
    """Read the observer position embedded by the repopulation writer."""
    base = "inputs/host/position_Earth"
    unit = u.Unit(_read_text_scalar(handle, f"{base}/unit", path))
    if unit != u.kpc:
        raise ValueError(
            f"Unit mismatch at {base + '/unit'!r} in {path}: expected kpc, "
            f"found {unit}."
        )

    values = []
    for component in range(3):
        location = f"{base}/value/item_{component}"
        if location not in handle:
            raise KeyError(f"Missing required dataset {location!r} in {path}")
        raw = np.asarray(handle[location][...])
        if raw.shape != ():
            raise ValueError(
                f"Expected one scalar at {location!r} in {path}; "
                f"found shape {raw.shape}."
            )
        value = float(raw.item())
        if not np.isfinite(value):
            raise ValueError(f"Non-finite observer component at {location!r} in {path}")
        values.append(value)

    position = np.asarray(values, dtype=np.float64)
    if position[0] <= 0.0 or not np.allclose(position[1:], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            "Masked-boost geometry requires the pipeline convention "
            f"position_Earth=(R0, 0, 0) with R0 > 0; found {position.tolist()} "
            f"in {path}."
        )
    return position


def validate_observer_consistency(
    reference: np.ndarray,
    candidate: np.ndarray,
    path: Path,
) -> None:
    if not np.allclose(candidate, reference, rtol=1.0e-12, atol=0.0):
        raise ValueError(
            f"Embedded observer position is inconsistent in {path}: "
            f"{candidate.tolist()}, expected {reference.tolist()}."
        )


def validate_mask_b_cut(mask_b_cut_deg: float) -> None:
    if not np.isfinite(mask_b_cut_deg) or not 0.0 <= mask_b_cut_deg < 90.0:
        raise ValueError("mask_b_cut_deg must be finite and satisfy 0 <= b_cut < 90")


def validate_smooth_rtol(smooth_rtol: float) -> None:
    if not np.isfinite(smooth_rtol) or not 0.0 < smooth_rtol < 1.0:
        raise ValueError("smooth_rtol must be finite and satisfy 0 < rtol < 1")


def _angular_unmasked_fraction(
    u: float,
    radius_kpc: float,
    observer_radius_kpc: float,
    sin_b_cut: float,
    tan_b_cut: float,
) -> float:
    """Azimuthal fraction outside the mask at fixed radius and |cos(theta)|.

    The observer is at (R0, 0, 0).  On a Galactocentric shell, let
    ``u=|z|/r`` and ``R=r*sqrt(1-u^2)``.  The unmasked condition is

        z^2 >= tan(b_cut)^2 * (R^2 + R0^2 - 2 R R0 cos(phi)).

    Solving it for phi leaves an analytic azimuthal fraction; only the u and
    Galactocentric-radius integrations remain numerical.
    """
    cylindrical_radius = radius_kpc * math.sqrt(max(0.0, 1.0 - u * u))
    z = radius_kpc * u
    denominator = 2.0 * cylindrical_radius * observer_radius_kpc

    if denominator == 0.0:
        distance = math.hypot(cylindrical_radius - observer_radius_kpc, z)
        return float(abs(z) >= distance * sin_b_cut)

    boundary = (
        cylindrical_radius**2
        + observer_radius_kpc**2
        - z**2 / tan_b_cut**2
    ) / denominator
    if boundary <= -1.0:
        return 1.0
    if boundary >= 1.0:
        return 0.0
    return math.acos(boundary) / math.pi


def smooth_host_luminosity_region(
    host: HostParameters,
    observer_radius_kpc: float,
    mask_b_cut_deg: float,
    *,
    keep_unmasked: bool = True,
    rtol: float = DEFAULT_SMOOTH_RTOL,
) -> np.longdouble:
    """Integrate original smooth-NFW luminosity in one angular region.

    The integration is performed in Galactocentric spherical coordinates, so
    the host truncation is exactly ``r <= Rvir``.  The radial integrand is
    ``r^2 rho(r)^2``; it is the same volume Jacobian represented by ``s^2`` in
    observer-centred coordinates, and is finite at the NFW origin.

    ``keep_unmasked=True`` integrates |b| >= b_cut.  False integrates the
    complementary band |b| < b_cut independently.
    """
    validate_mask_b_cut(mask_b_cut_deg)
    validate_smooth_rtol(rtol)
    if not np.isfinite(observer_radius_kpc) or observer_radius_kpc <= 0.0:
        raise ValueError("observer_radius_kpc must be finite and positive")
    if observer_radius_kpc >= host.r200_kpc:
        raise ValueError("observer must lie inside the smooth host truncation radius")

    rho0 = float(host.rho0_msun_kpc3)
    rs = float(host.rs_kpc)
    rvir = float(host.r200_kpc)

    if mask_b_cut_deg == 0.0:
        angular_solid_angle = 4.0 * math.pi if keep_unmasked else 0.0

        def radial_integrand_full_sky(radius: float) -> float:
            return (
                angular_solid_angle
                * rho0**2
                * rs**2
                / (1.0 + radius / rs) ** 4
            )

        value, _ = quad(
            radial_integrand_full_sky,
            0.0,
            rvir,
            epsabs=0.0 if angular_solid_angle else 1.0e-300,
            epsrel=rtol,
            limit=200,
        )
        return ACCUMULATION_DTYPE(value)

    b_cut_rad = math.radians(mask_b_cut_deg)
    sin_b_cut = math.sin(b_cut_rad)
    tan_b_cut = math.tan(b_cut_rad)
    angular_tolerance = max(0.1 * rtol, 1.0e-12)
    fully_masked_radius = observer_radius_kpc * sin_b_cut

    def unmasked_solid_angle(radius: float) -> float:
        if radius < fully_masked_radius:
            return 0.0
        integral, _ = quad(
            _angular_unmasked_fraction,
            0.0,
            1.0,
            args=(
                radius,
                observer_radius_kpc,
                sin_b_cut,
                tan_b_cut,
            ),
            epsabs=angular_tolerance,
            epsrel=angular_tolerance,
            limit=200,
        )
        return 4.0 * math.pi * integral

    def radial_integrand(radius: float) -> float:
        omega_unmasked = unmasked_solid_angle(radius)
        omega = omega_unmasked if keep_unmasked else 4.0 * math.pi - omega_unmasked
        radial_density = rho0**2 * rs**2 / (1.0 + radius / rs) ** 4
        return radial_density * omega

    points = sorted(
        point
        for point in {fully_masked_radius, observer_radius_kpc}
        if 0.0 < point < rvir
    )
    value, _ = quad(
        radial_integrand,
        0.0,
        rvir,
        points=points,
        epsabs=0.0,
        epsrel=rtol,
        limit=200,
    )
    return ACCUMULATION_DTYPE(value)


def validate_catalogue_metadata(
    handle: h5py.File,
    path: Path,
) -> tuple[h5py.Dataset, dict[str, int], HostParameters, np.ndarray]:
    """Validate metadata without reading catalogue rows."""
    if DATASET_NAME not in handle:
        raise KeyError(f"Missing required dataset {DATASET_NAME!r} in {path}")
    dataset = handle[DATASET_NAME]
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2:
        raise ValueError(f"Expected a two-dimensional dataset at {DATASET_NAME!r} in {path}")
    if dataset.shape[0] <= 0:
        raise ValueError(f"Catalogue contains no saved subhalos: {path}")
    if "column_names" not in dataset.attrs or "units" not in dataset.attrs:
        raise KeyError(f"Missing column_names or units metadata on {DATASET_NAME!r} in {path}")

    column_names = _decode_string_array(dataset.attrs["column_names"])
    units = _decode_string_array(dataset.attrs["units"])
    if len(column_names) != dataset.shape[1] or len(units) != dataset.shape[1]:
        raise ValueError(f"Column metadata does not match dataset shape in {path}")
    if len(set(column_names)) != len(column_names):
        raise ValueError(f"Duplicate column names in {path}: {column_names}")

    missing = [name for name in REQUIRED_COLUMNS if name not in column_names]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {', '.join(missing)}")
    indices = {name: column_names.index(name) for name in REQUIRED_COLUMNS}
    for name, expected in EXPECTED_COLUMN_UNITS.items():
        try:
            actual = u.Unit(units[indices[name]])
        except ValueError as error:
            raise ValueError(f"Invalid unit for column {name!r} in {path}") from error
        if actual != expected:
            raise ValueError(
                f"Unit mismatch for column {name!r} in {path}: "
                f"expected {expected}, found {actual}."
            )

    group = handle[ITERATION_GROUP]
    if "n_saved" not in group.attrs:
        raise KeyError(f"Missing n_saved attribute on {ITERATION_GROUP!r} in {path}")
    n_saved = int(group.attrs["n_saved"])
    if n_saved != dataset.shape[0]:
        raise ValueError(
            f"Incomplete or truncated catalogue {path}: n_saved={n_saved}, "
            f"but {DATASET_NAME} has {dataset.shape[0]} rows."
        )

    return (
        dataset,
        indices,
        read_host_parameters(handle, path),
        read_observer_position(handle, path),
    )


def inspect_catalogue(path: Path) -> tuple[HostParameters, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Input HDF5 catalogue not found: {path}")
    with h5py.File(path, "r") as handle:
        _, _, host, observer = validate_catalogue_metadata(handle, path)
    return host, observer


def _validate_chunk(
    d_earth: np.ndarray,
    zearth: np.ndarray,
    rs: np.ndarray,
    rho_s: np.ndarray,
    path: Path,
    start: int,
) -> None:
    arrays = {"D_Earth": d_earth, "Zearth": zearth, "r_s": rs, "rho_s": rho_s}
    for name, array in arrays.items():
        invalid = ~np.isfinite(array)
        if name != "Zearth":
            invalid |= array <= 0.0
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(f"Invalid {name} at row {start + index} in {path}")
    inconsistent = np.abs(zearth) > d_earth * (1.0 + 1.0e-12)
    if np.any(inconsistent):
        index = int(np.flatnonzero(inconsistent)[0])
        raise ValueError(
            f"|Zearth| exceeds D_Earth at row {start + index} in {path}"
        )


def scan_catalogue(
    path: Path,
    repop_id: int,
    chunk_size: int,
    mask_b_cut_deg: float,
    l_mw_unmasked: np.longdouble | None = None,
    smooth_rtol: float = DEFAULT_SMOOTH_RTOL,
) -> tuple[MaskedBoostResult, HostParameters, np.ndarray]:
    """Scan one catalogue in bounded-memory chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    validate_mask_b_cut(mask_b_cut_deg)
    if not path.is_file():
        raise FileNotFoundError(f"Input HDF5 catalogue not found: {path}")

    total_l = ACCUMULATION_DTYPE(0.0)
    unmasked_l = ACCUMULATION_DTYPE(0.0)
    n_unmasked = 0
    sin_b_cut = math.sin(math.radians(mask_b_cut_deg))

    with h5py.File(path, "r") as handle:
        dataset, indices, host, observer = validate_catalogue_metadata(handle, path)
        n_rows = int(dataset.shape[0])

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            d_earth = np.asarray(
                dataset[start:stop, indices["D_Earth"]], dtype=np.float64
            )
            zearth = np.asarray(
                dataset[start:stop, indices["Zearth"]], dtype=np.float64
            )
            rs = np.asarray(
                dataset[start:stop, indices["r_s"]], dtype=ACCUMULATION_DTYPE
            )
            rho_s = np.asarray(
                dataset[start:stop, indices["rho_s"]], dtype=ACCUMULATION_DTYPE
            )
            _validate_chunk(d_earth, zearth, rs, rho_s, path, start)

            luminosity = SUBHALO_LUMINOSITY_COEFFICIENT * rho_s**2 * rs**3
            unmasked = np.abs(zearth) >= d_earth * sin_b_cut
            total_l += np.sum(luminosity, dtype=ACCUMULATION_DTYPE)
            unmasked_l += np.sum(luminosity[unmasked], dtype=ACCUMULATION_DTYPE)
            n_unmasked += int(np.count_nonzero(unmasked))

    l_mw_total = host_luminosity(host)
    if l_mw_unmasked is None:
        l_mw_unmasked = smooth_host_luminosity_region(
            host,
            float(observer[0]),
            mask_b_cut_deg,
            rtol=smooth_rtol,
        )
    l_mw_unmasked = ACCUMULATION_DTYPE(l_mw_unmasked)

    if not np.isfinite(total_l) or total_l <= 0.0:
        raise ValueError(f"Invalid total subhalo luminosity in {path}: {total_l!r}")
    if not np.isfinite(unmasked_l) or unmasked_l < 0.0 or unmasked_l > total_l:
        raise ValueError(f"Invalid unmasked subhalo luminosity in {path}: {unmasked_l!r}")
    if not 0 <= n_unmasked <= n_rows:
        raise ValueError(f"Invalid unmasked subhalo count in {path}: {n_unmasked}")
    if (
        not np.isfinite(l_mw_unmasked)
        or l_mw_unmasked <= 0.0
        or l_mw_unmasked > l_mw_total * ACCUMULATION_DTYPE(1.0 + 1.0e-12)
    ):
        raise ValueError(f"Invalid unmasked smooth luminosity: {l_mw_unmasked!r}")

    result = MaskedBoostResult(
        repop_id=repop_id,
        boost_factor_masked=unmasked_l / l_mw_unmasked,
        boost_factor_global=total_l / l_mw_total,
        l_sub_unmasked=unmasked_l,
        l_sub_total=total_l,
        l_mw_unmasked=l_mw_unmasked,
        l_mw_total=l_mw_total,
        n_subhalos_unmasked=n_unmasked,
        n_subhalos_total=n_rows,
        mask_b_cut_deg=mask_b_cut_deg,
    )
    return result, host, observer


def result_to_row(result: MaskedBoostResult) -> dict[str, str]:
    return {
        "repop_id": str(result.repop_id),
        "boost_factor_masked": _format_extended(result.boost_factor_masked),
        "boost_factor_global": _format_extended(result.boost_factor_global),
        "L_sub_unmasked": _format_extended(result.l_sub_unmasked),
        "L_sub_total": _format_extended(result.l_sub_total),
        "L_MW_unmasked": _format_extended(result.l_mw_unmasked),
        "L_MW_total": _format_extended(result.l_mw_total),
        "n_subhalos_unmasked": str(result.n_subhalos_unmasked),
        "n_subhalos_total": str(result.n_subhalos_total),
        "mask_b_cut_deg": np.format_float_positional(
            result.mask_b_cut_deg, unique=True, trim="-"
        ),
    }


def _validate_csv_row(row: dict[str, str], path: Path, line_number: int) -> int:
    try:
        repop_id = int(row["repop_id"])
        n_unmasked = int(row["n_subhalos_unmasked"])
        n_total = int(row["n_subhalos_total"])
        mask_b_cut = float(row["mask_b_cut_deg"])
        values = {
            name: float(row[name])
            for name in CSV_FIELDS
            if name not in {
                "repop_id",
                "n_subhalos_unmasked",
                "n_subhalos_total",
                "mask_b_cut_deg",
            }
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid CSV row at {path}:{line_number}: {row}") from error

    validate_mask_b_cut(mask_b_cut)
    if repop_id < 0 or n_total <= 0 or not 0 <= n_unmasked <= n_total:
        raise ValueError(f"Invalid CSV row at {path}:{line_number}: {row}")
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError(f"Invalid CSV row at {path}:{line_number}: {row}")
    for name in ("L_sub_total", "L_MW_unmasked", "L_MW_total"):
        if values[name] <= 0.0:
            raise ValueError(f"Invalid CSV row at {path}:{line_number}: {row}")
    if values["L_sub_unmasked"] > values["L_sub_total"]:
        raise ValueError(f"Invalid CSV row at {path}:{line_number}: {row}")
    if values["L_MW_unmasked"] > values["L_MW_total"] * (1.0 + 1.0e-12):
        raise ValueError(f"Invalid CSV row at {path}:{line_number}: {row}")
    return repop_id


def load_existing_rows(path: Path, mask_b_cut_deg: float) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                f"Unexpected columns in {path}: {reader.fieldnames}; "
                f"expected {list(CSV_FIELDS)}."
            )
        for line_number, row in enumerate(reader, start=2):
            repop_id = _validate_csv_row(row, path, line_number)
            if not np.isclose(
                float(row["mask_b_cut_deg"]), mask_b_cut_deg, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"Mask mismatch in {path}:{line_number}: "
                    f"found {row['mask_b_cut_deg']}, requested {mask_b_cut_deg}."
                )
            if repop_id in rows:
                raise ValueError(f"Duplicate repop_id {repop_id} in {path}")
            rows[repop_id] = row
    return rows


def write_rows_atomic(path: Path, rows: dict[int, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for repop_id in sorted(rows):
                writer.writerow(rows[repop_id])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def run_scan(
    *,
    root: Path,
    scenario: str,
    start_id: int,
    stop_id: int,
    chunk_size: int,
    mask_b_cut_deg: float,
    smooth_rtol: float,
    output: Path,
    resume: bool,
) -> dict[int, dict[str, str]]:
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"scenario must be one of {VALID_SCENARIOS}")
    if start_id < 0 or stop_id <= start_id:
        raise ValueError("Require 0 <= start_id < stop_id (stop_id is exclusive).")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    validate_mask_b_cut(mask_b_cut_deg)
    validate_smooth_rtol(smooth_rtol)
    if output.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {output}. Use --resume to preserve its rows."
        )

    rows = load_existing_rows(output, mask_b_cut_deg) if resume else {}
    first_path = catalogue_path(root, start_id, scenario)
    reference_host, reference_observer = inspect_catalogue(first_path)
    l_mw_unmasked = smooth_host_luminosity_region(
        reference_host,
        float(reference_observer[0]),
        mask_b_cut_deg,
        rtol=smooth_rtol,
    )

    total_requested = stop_id - start_id
    for sequence, repop_id in enumerate(range(start_id, stop_id), start=1):
        path = catalogue_path(root, repop_id, scenario)
        if repop_id in rows:
            host, observer = inspect_catalogue(path)
            status = "resume"
        else:
            result, host, observer = scan_catalogue(
                path,
                repop_id,
                chunk_size,
                mask_b_cut_deg,
                l_mw_unmasked=l_mw_unmasked,
                smooth_rtol=smooth_rtol,
            )
            status = "saved"

        validate_host_consistency(reference_host, host, path)
        validate_observer_consistency(reference_observer, observer, path)
        if repop_id in rows:
            recorded_l_mw = float(rows[repop_id]["L_MW_unmasked"])
            if not np.isclose(
                recorded_l_mw, float(l_mw_unmasked), rtol=1.0e-12, atol=0.0
            ):
                raise ValueError(
                    f"Stored L_MW_unmasked for repop_id={repop_id} is inconsistent "
                    f"with the requested geometry: {recorded_l_mw!r} != "
                    f"{float(l_mw_unmasked)!r}."
                )
        else:
            rows[repop_id] = result_to_row(result)
            write_rows_atomic(output, rows)
        print(
            f"[{sequence:04d}/{total_requested:04d}] repop_{repop_id:04d} "
            f"{scenario}: {status}",
            flush=True,
        )

    return rows


def main() -> None:
    args = parse_args()
    output = args.output if args.output is not None else default_output_path(args.scenario)
    run_scan(
        root=args.root,
        scenario=args.scenario,
        start_id=args.start_id,
        stop_id=args.stop_id,
        chunk_size=args.chunk_size,
        mask_b_cut_deg=args.mask_b_cut_deg,
        smooth_rtol=args.smooth_rtol,
        output=output,
        resume=args.resume,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
