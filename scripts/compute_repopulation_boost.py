#!/usr/bin/env python3
"""Compute the intrinsic annihilation boost of saved repopulations.

For every saved row in an original repopulated HDF5 catalogue, this product
uses the stored NFW characteristic density and scale radius to compute

    L_i = (7 pi / 6) rho_s^2 r_s^3.

The per-repopulation boost is ``sum(L_i) / L_MW0``, where ``L_MW0`` is the
luminosity of the original smooth NFW host.  The host parameters are read from
the catalogue's embedded inputs and are never taken from the renormalized
CLUMPY configuration.

This is an intrinsic, per-repopulation quantity.  It is independent of the
observer position, NSIDE, pointlike/extended classification, and map catalogue
cuts.  It includes all SAVED subhalos in ``iteration_0/data``; physical objects
removed by the repopulation model before HDF5 storage are outside its scope.
Only the required columns are read, in configurable chunks.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from astropy import units as u


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_START_ID = 0
DEFAULT_STOP_ID = 500
VALID_SCENARIOS = ("fragile", "resilient")
DATASET_NAME = "iteration_0/data"
ITERATION_GROUP = "iteration_0"

CSV_FIELDS = (
    "repop_id",
    "boost_factor",
    "L_sub_total",
    "n_subhalos",
    "sum_Js",
    "max_Js",
    "max_L_subhalo",
    "frac_Lmax",
    "vmax_at_max_Js",
    "vmax_at_max_L",
)

REQUIRED_COLUMNS = ("Js", "Vmax", "r_s", "rho_s")
EXPECTED_COLUMN_UNITS = {
    "Js": u.GeV**2 / u.cm**5,
    "Vmax": u.km / u.s,
    "r_s": u.kpc,
    "rho_s": u.Msun / u.kpc**3,
}

HOST_DATASETS = {
    "rho0": "inputs/host/rho_0/value",
    "rho0_unit": "inputs/host/rho_0/unit",
    "rs": "inputs/host/r_s/value",
    "rs_unit": "inputs/host/r_s/unit",
    "r200": "inputs/host/R_vir/value",
    "r200_unit": "inputs/host/R_vir/unit",
    "delta": "inputs/cosmo_constants/delta_overdensity",
}

ACCUMULATION_DTYPE = np.longdouble
PI = np.arccos(ACCUMULATION_DTYPE(-1.0))
SUBHALO_LUMINOSITY_COEFFICIENT = (
    ACCUMULATION_DTYPE(7.0) * PI / ACCUMULATION_DTYPE(6.0)
)


@dataclass(frozen=True)
class HostParameters:
    """Original smooth-host parameters in native catalogue units."""

    rho0_msun_kpc3: float
    rs_kpc: float
    r200_kpc: float
    delta_overdensity: float


@dataclass(frozen=True)
class BoostResult:
    """One row of the repopulation boost product."""

    repop_id: int
    boost_factor: np.longdouble
    l_sub_total: np.longdouble
    n_subhalos: int
    sum_js: np.longdouble
    max_js: float
    max_l_subhalo: np.longdouble
    frac_lmax: np.longdouble
    vmax_at_max_js: float
    vmax_at_max_l: float


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
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV. By default, writes outputs/repopulation_boost/"
            "substructure_boost_<scenario>_0500.csv."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve valid existing rows and compute only missing IDs.",
    )
    return parser.parse_args()


def catalogue_path(root: Path, repop_id: int, scenario: str) -> Path:
    return (
        root
        / f"repop_{repop_id:04d}"
        / f"fullrepop_hydro_{scenario}.h5"
    )


def default_output_path(scenario: str) -> Path:
    return (
        REPOSITORY_ROOT
        / "outputs"
        / "repopulation_boost"
        / f"substructure_boost_{scenario}_0500.csv"
    )


def _decode_text(value: object, *, location: str, path: Path) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise ValueError(
        f"Expected text at {location!r} in {path}; found {type(value).__name__}."
    )


def _read_scalar(handle: h5py.File, location: str, path: Path) -> float:
    if location not in handle:
        raise KeyError(f"Missing required dataset {location!r} in {path}")
    dataset = handle[location]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Expected a dataset at {location!r} in {path}")
    value = np.asarray(dataset[...])
    if value.shape != ():
        raise ValueError(
            f"Expected one scalar at {location!r} in {path}; found shape {value.shape}."
        )
    try:
        scalar = float(value.item())
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid scalar at {location!r} in {path}") from error
    if not np.isfinite(scalar):
        raise ValueError(f"Non-finite scalar at {location!r} in {path}: {scalar!r}")
    return scalar


def _read_unit(handle: h5py.File, location: str, path: Path) -> u.UnitBase:
    if location not in handle:
        raise KeyError(f"Missing required unit dataset {location!r} in {path}")
    dataset = handle[location]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Expected a dataset at {location!r} in {path}")
    raw = np.asarray(dataset[...])
    if raw.shape != ():
        raise ValueError(
            f"Expected one unit string at {location!r} in {path}; found shape {raw.shape}."
        )
    text = _decode_text(raw.item(), location=location, path=path)
    try:
        return u.Unit(text)
    except ValueError as error:
        raise ValueError(f"Invalid unit {text!r} at {location!r} in {path}") from error


def read_host_parameters(handle: h5py.File, path: Path) -> HostParameters:
    """Read and validate the original host parameters embedded in one HDF5."""
    rho0 = _read_scalar(handle, HOST_DATASETS["rho0"], path)
    rs = _read_scalar(handle, HOST_DATASETS["rs"], path)
    r200 = _read_scalar(handle, HOST_DATASETS["r200"], path)
    delta = _read_scalar(handle, HOST_DATASETS["delta"], path)

    rho0_unit = _read_unit(handle, HOST_DATASETS["rho0_unit"], path)
    rs_unit = _read_unit(handle, HOST_DATASETS["rs_unit"], path)
    r200_unit = _read_unit(handle, HOST_DATASETS["r200_unit"], path)

    expected_units = (
        (HOST_DATASETS["rho0_unit"], rho0_unit, u.Msun / u.kpc**3),
        (HOST_DATASETS["rs_unit"], rs_unit, u.kpc),
        (HOST_DATASETS["r200_unit"], r200_unit, u.kpc),
    )
    for location, actual, expected in expected_units:
        if actual != expected:
            raise ValueError(
                f"Unit mismatch at {location!r} in {path}: expected {expected}, "
                f"found {actual}."
            )

    finite_values = {
        "rho_0": rho0,
        "r_s": rs,
        "R_vir": r200,
        "delta_overdensity": delta,
    }
    for name, value in finite_values.items():
        if not np.isfinite(value):
            raise ValueError(
                f"Embedded host {name} must be finite in {path}: {value!r}"
            )

    positive_values = {"rho_0": rho0, "r_s": rs, "R_vir": r200}
    for name, value in positive_values.items():
        if value <= 0.0:
            raise ValueError(f"Embedded host {name} must be positive in {path}: {value!r}")
    if not np.isclose(delta, 200.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"Embedded delta_overdensity must be 200 in {path}; found {delta!r}."
        )

    return HostParameters(rho0, rs, r200, delta)


def host_luminosity(host: HostParameters) -> np.longdouble:
    """Return original smooth-host NFW luminosity in Msun^2 kpc^-3."""
    rho0 = ACCUMULATION_DTYPE(host.rho0_msun_kpc3)
    rs = ACCUMULATION_DTYPE(host.rs_kpc)
    concentration = ACCUMULATION_DTYPE(host.r200_kpc) / rs
    radial_factor = ACCUMULATION_DTYPE(1.0) - (
        ACCUMULATION_DTYPE(1.0) + concentration
    ) ** ACCUMULATION_DTYPE(-3.0)
    return (
        ACCUMULATION_DTYPE(4.0)
        * PI
        / ACCUMULATION_DTYPE(3.0)
        * rho0**2
        * rs**3
        * radial_factor
    )


def validate_host_consistency(
    reference: HostParameters,
    candidate: HostParameters,
    path: Path,
) -> None:
    for field in HostParameters.__dataclass_fields__:
        expected = getattr(reference, field)
        actual = getattr(candidate, field)
        if not np.isclose(actual, expected, rtol=1.0e-12, atol=0.0):
            raise ValueError(
                f"Embedded host parameters are inconsistent in {path}: "
                f"{field}={actual!r}, expected {expected!r}."
            )


def _decode_string_array(values: Iterable[object]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def validate_catalogue_metadata(
    handle: h5py.File,
    path: Path,
) -> tuple[h5py.Dataset, dict[str, int], HostParameters]:
    """Validate inexpensive structure/metadata and return the open data table."""
    if DATASET_NAME not in handle:
        raise KeyError(f"Missing required dataset {DATASET_NAME!r} in {path}")
    dataset = handle[DATASET_NAME]
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional dataset at {DATASET_NAME!r} in {path}."
        )
    if dataset.shape[0] <= 0:
        raise ValueError(f"Catalogue contains no saved subhalos: {path}")
    if "column_names" not in dataset.attrs:
        raise KeyError(f"Missing column_names attribute on {DATASET_NAME!r} in {path}")
    if "units" not in dataset.attrs:
        raise KeyError(f"Missing units attribute on {DATASET_NAME!r} in {path}")

    column_names = _decode_string_array(dataset.attrs["column_names"])
    units = _decode_string_array(dataset.attrs["units"])
    if len(column_names) != dataset.shape[1]:
        raise ValueError(
            f"Column-name metadata mismatch in {path}: {len(column_names)} names "
            f"for {dataset.shape[1]} columns."
        )
    if len(units) != dataset.shape[1]:
        raise ValueError(
            f"Unit metadata mismatch in {path}: {len(units)} units for "
            f"{dataset.shape[1]} columns."
        )
    if len(set(column_names)) != len(column_names):
        raise ValueError(f"Duplicate column names in {path}: {column_names}")

    missing = [name for name in REQUIRED_COLUMNS if name not in column_names]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {', '.join(missing)}")

    indices = {name: column_names.index(name) for name in REQUIRED_COLUMNS}
    for name, expected in EXPECTED_COLUMN_UNITS.items():
        raw_unit = units[indices[name]]
        try:
            actual = u.Unit(raw_unit)
        except ValueError as error:
            raise ValueError(
                f"Invalid unit {raw_unit!r} for column {name!r} in {path}"
            ) from error
        if actual != expected:
            raise ValueError(
                f"Unit mismatch for column {name!r} in {path}: expected "
                f"{expected}, found {actual}."
            )

    group = handle[ITERATION_GROUP]
    if "n_saved" not in group.attrs:
        raise KeyError(f"Missing n_saved attribute on {ITERATION_GROUP!r} in {path}")
    try:
        n_saved = int(group.attrs["n_saved"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid n_saved attribute in {path}") from error
    if n_saved != dataset.shape[0]:
        raise ValueError(
            f"Incomplete or truncated catalogue {path}: n_saved={n_saved}, "
            f"but {DATASET_NAME} has {dataset.shape[0]} rows."
        )

    return dataset, indices, read_host_parameters(handle, path)


def inspect_catalogue(path: Path) -> HostParameters:
    """Validate catalogue metadata without reading its table rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Input HDF5 catalogue not found: {path}")
    with h5py.File(path, "r") as handle:
        _, _, host = validate_catalogue_metadata(handle, path)
    return host


def _validate_chunk(values: dict[str, np.ndarray], path: Path, start: int) -> None:
    for name, array in values.items():
        invalid = ~np.isfinite(array) | (array <= 0.0)
        if np.any(invalid):
            local_index = int(np.flatnonzero(invalid)[0])
            value = array[local_index]
            raise ValueError(
                f"Invalid {name} at row {start + local_index} in {path}: "
                f"expected a finite positive value, found {value!r}."
            )


def scan_catalogue(
    path: Path,
    repop_id: int,
    chunk_size: int,
) -> tuple[BoostResult, HostParameters]:
    """Scan one catalogue in chunks and return its boost-product row."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not path.is_file():
        raise FileNotFoundError(f"Input HDF5 catalogue not found: {path}")

    total_l = ACCUMULATION_DTYPE(0.0)
    total_js = ACCUMULATION_DTYPE(0.0)
    max_js = -math.inf
    max_l = ACCUMULATION_DTYPE(-np.inf)
    vmax_at_max_js = math.nan
    vmax_at_max_l = math.nan

    with h5py.File(path, "r") as handle:
        dataset, indices, host = validate_catalogue_metadata(handle, path)
        n_rows = int(dataset.shape[0])

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            js = np.asarray(dataset[start:stop, indices["Js"]], dtype=np.float64)
            vmax = np.asarray(dataset[start:stop, indices["Vmax"]], dtype=np.float64)
            rs = np.asarray(
                dataset[start:stop, indices["r_s"]], dtype=ACCUMULATION_DTYPE
            )
            rho_s = np.asarray(
                dataset[start:stop, indices["rho_s"]], dtype=ACCUMULATION_DTYPE
            )
            _validate_chunk(
                {"Js": js, "Vmax": vmax, "r_s": rs, "rho_s": rho_s},
                path,
                start,
            )

            luminosity = SUBHALO_LUMINOSITY_COEFFICIENT * rho_s**2 * rs**3
            total_js += np.sum(js, dtype=ACCUMULATION_DTYPE)
            total_l += np.sum(luminosity, dtype=ACCUMULATION_DTYPE)

            local_js_index = int(np.argmax(js))
            local_js_max = float(js[local_js_index])
            if local_js_max > max_js:
                max_js = local_js_max
                vmax_at_max_js = float(vmax[local_js_index])

            local_l_index = int(np.argmax(luminosity))
            local_l_max = luminosity[local_l_index]
            if local_l_max > max_l:
                max_l = local_l_max
                vmax_at_max_l = float(vmax[local_l_index])

    l_mw0 = host_luminosity(host)
    if not np.isfinite(total_l) or total_l <= 0.0:
        raise ValueError(f"Invalid accumulated subhalo luminosity in {path}: {total_l!r}")
    if not np.isfinite(total_js) or total_js <= 0.0:
        raise ValueError(f"Invalid accumulated Js in {path}: {total_js!r}")
    if not np.isfinite(l_mw0) or l_mw0 <= 0.0:
        raise ValueError(f"Invalid original host luminosity in {path}: {l_mw0!r}")

    return (
        BoostResult(
            repop_id=repop_id,
            boost_factor=total_l / l_mw0,
            l_sub_total=total_l,
            n_subhalos=n_rows,
            sum_js=total_js,
            max_js=max_js,
            max_l_subhalo=max_l,
            frac_lmax=max_l / total_l,
            vmax_at_max_js=vmax_at_max_js,
            vmax_at_max_l=vmax_at_max_l,
        ),
        host,
    )


def _format_extended(value: np.longdouble | float) -> str:
    precision = int(np.finfo(ACCUMULATION_DTYPE).precision)
    return np.format_float_scientific(
        ACCUMULATION_DTYPE(value),
        precision=precision,
        unique=False,
        trim="k",
    )


def result_to_row(result: BoostResult) -> dict[str, str]:
    return {
        "repop_id": str(result.repop_id),
        "boost_factor": _format_extended(result.boost_factor),
        "L_sub_total": _format_extended(result.l_sub_total),
        "n_subhalos": str(result.n_subhalos),
        "sum_Js": _format_extended(result.sum_js),
        "max_Js": _format_extended(result.max_js),
        "max_L_subhalo": _format_extended(result.max_l_subhalo),
        "frac_Lmax": _format_extended(result.frac_lmax),
        "vmax_at_max_Js": _format_extended(result.vmax_at_max_js),
        "vmax_at_max_L": _format_extended(result.vmax_at_max_l),
    }


def _validate_csv_row(row: dict[str, str], path: Path, line_number: int) -> int:
    try:
        repop_id = int(row["repop_id"])
        n_subhalos = int(row["n_subhalos"])
        values = {
            name: float(row[name])
            for name in CSV_FIELDS
            if name not in {"repop_id", "n_subhalos"}
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid CSV row at {path}:{line_number}: {row}") from error
    if repop_id < 0 or n_subhalos <= 0:
        raise ValueError(f"Invalid CSV row at {path}:{line_number}: {row}")
    if any(not np.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError(f"Invalid CSV row at {path}:{line_number}: {row}")
    if values["frac_Lmax"] > 1.0:
        raise ValueError(f"Invalid frac_Lmax at {path}:{line_number}: {row}")
    return repop_id


def load_existing_rows(path: Path) -> dict[int, dict[str, str]]:
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
            if repop_id in rows:
                raise ValueError(f"Duplicate repop_id {repop_id} in {path}")
            rows[repop_id] = row
    return rows


def write_rows_atomic(path: Path, rows: dict[int, dict[str, str]]) -> None:
    """Write sorted product rows through a same-directory temporary file."""
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
    output: Path,
    resume: bool,
) -> dict[int, dict[str, str]]:
    """Process the half-open repopulation range ``[start_id, stop_id)``."""
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"scenario must be one of {VALID_SCENARIOS}")
    if start_id < 0 or stop_id <= start_id:
        raise ValueError("Require 0 <= start_id < stop_id (stop_id is exclusive).")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if output.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {output}. Use --resume to preserve its rows."
        )

    rows = load_existing_rows(output) if resume else {}
    if resume and list(rows) != sorted(rows):
        write_rows_atomic(output, rows)
    reference_host: HostParameters | None = None
    total_requested = stop_id - start_id

    for sequence, repop_id in enumerate(range(start_id, stop_id), start=1):
        path = catalogue_path(root, repop_id, scenario)
        already_complete = repop_id in rows
        if already_complete:
            host = inspect_catalogue(path)
            status = "resume"
        else:
            result, host = scan_catalogue(path, repop_id, chunk_size)
            status = "saved"

        if reference_host is None:
            reference_host = host
        else:
            validate_host_consistency(reference_host, host, path)

        if not already_complete:
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
        output=output,
        resume=args.resume,
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
