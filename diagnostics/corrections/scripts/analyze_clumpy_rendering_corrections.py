#!/usr/bin/env python3
"""Validate and analyse raw/corrected CLUMPY halo rendering.

The catalogue and CLUMPY products deliberately have independent roots.  The
catalogue is the authoritative source of ``Js`` and ``theta_s``; CLUMPY list
files are used only to define the rendered extended-halo sample and its order.
All ensemble statistics are formed from per-repopulation statistics so that
each repopulation has equal weight.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.io.fits.verify import VerifyError


SCENARIOS = ("fragile", "resilient")
SCENARIO_COLORS = {
    "fragile": "#EE3377",
    "resilient": "#009988",
}
DEFAULT_NSIDE = 2048
DEFAULT_X_BINS = 24
DEFAULT_CDF_POINTS = 512
HALO_NAME_PATTERN = re.compile(r"^hydro_res_(0*[1-9][0-9]*)$")
LOG_HEADER = (
    "HALO_INDEX",
    "NAME",
    "TYPE",
    "J_RENDERED",
    "NPIX_TOUCHED",
)


class ProductValidationError(RuntimeError):
    """A requested production product is absent, incomplete, or inconsistent."""


@dataclass(frozen=True)
class ListRow:
    name: str
    halo_type: str


@dataclass(frozen=True)
class RenderedRow:
    halo_index: int
    name: str
    halo_type: str
    j_rendered: float


@dataclass(frozen=True)
class CasePaths:
    catalog: Path
    raw_list: Path
    corrected_list: Path
    raw_log: Path
    corrected_log: Path
    total_fits: Path


@dataclass
class ValidationRecord:
    scenario: str
    repop_id: int
    catalog_found: bool = False
    catalog_valid: bool = False
    raw_list_found: bool = False
    corrected_list_found: bool = False
    raw_log_found: bool = False
    corrected_log_found: bool = False
    total_fits_valid: bool = False
    raw_list_matching_valid: bool = False
    corrected_list_matching_valid: bool = False
    raw_corrected_sequence_valid: bool = False
    hdf5_matching_valid: bool = False
    n_matched: int = 0
    usable: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def exclusion_reason(self) -> str:
        return "" if self.usable else "; ".join(dict.fromkeys(self.reasons))


@dataclass(frozen=True)
class CaseData:
    scenario: str
    repop_id: int
    names: tuple[str, ...]
    j_catalog: np.ndarray
    theta_s_deg: np.ndarray
    j_rendered_raw: np.ndarray
    j_rendered_corr: np.ndarray
    x: np.ndarray
    r_raw: np.ndarray
    r_corr: np.ndarray
    delta_raw: np.ndarray
    delta_corr: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyse raw/corrected CLUMPY halo rendering with "
            "equal statistical weight per repopulation."
        )
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        required=True,
        help="Root containing repop_XXXX/fullrepop_hydro_<scenario>.h5.",
    )
    parser.add_argument(
        "--clumpy-root",
        type=Path,
        required=True,
        help="Root containing <scenario>/lists and <scenario>/outputs.",
    )
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "both"),
        required=True,
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--repop-ids",
        nargs="+",
        help="Explicit IDs, separated by spaces and/or commas.",
    )
    selection.add_argument(
        "--start-id",
        type=int,
        help="First ID in a half-open [start-id, stop-id) range.",
    )
    parser.add_argument(
        "--stop-id",
        type=int,
        help="Exclusive end of the range selected with --start-id.",
    )
    parser.add_argument("--nside", type=int, default=DEFAULT_NSIDE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/corrections/outputs"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate requested cases without writing summaries or plots.",
    )
    parser.add_argument("--x-bins", type=int, default=DEFAULT_X_BINS)
    parser.add_argument("--cdf-points", type=int, default=DEFAULT_CDF_POINTS)
    return parser.parse_args(argv)


def parse_repop_ids(args: argparse.Namespace) -> list[int]:
    if args.repop_ids is not None:
        if args.stop_id is not None:
            raise ValueError("--stop-id cannot be combined with --repop-ids.")
        tokens: list[str] = []
        for item in args.repop_ids:
            tokens.extend(part.strip() for part in item.split(","))
        try:
            ids = [int(token) for token in tokens if token]
        except ValueError as error:
            raise ValueError("--repop-ids must contain integers.") from error
        if not ids:
            raise ValueError("--repop-ids must contain at least one ID.")
        if len(ids) != len(set(ids)):
            raise ValueError("--repop-ids must not contain duplicates.")
    else:
        if args.stop_id is None:
            raise ValueError("--stop-id is required with --start-id.")
        if args.start_id >= args.stop_id:
            raise ValueError("Require --start-id < --stop-id.")
        ids = list(range(args.start_id, args.stop_id))

    if any(value < 0 for value in ids):
        raise ValueError("Repopulation IDs must be non-negative.")
    return ids


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.nside <= 0 or (args.nside & (args.nside - 1)) != 0:
        raise ValueError("--nside must be a positive power of two.")
    if args.x_bins <= 0:
        raise ValueError("--x-bins must be positive.")
    if args.cdf_points < 2:
        raise ValueError("--cdf-points must be at least 2.")


def scenarios_requested(value: str) -> tuple[str, ...]:
    return SCENARIOS if value == "both" else (value,)


def clumpy_basename(nside: int) -> str:
    return f"annihil_gal2D_LOS0_0_FOV360x180_nside{nside}"


def build_case_paths(
    catalog_root: Path,
    clumpy_root: Path,
    scenario: str,
    repop_id: int,
    nside: int,
) -> CasePaths:
    tag = f"repop_{repop_id:04d}"
    run_tag = f"{tag}_nside{nside}"
    basename = clumpy_basename(nside)
    scenario_root = clumpy_root / scenario
    return CasePaths(
        catalog=(
            catalog_root / tag / f"fullrepop_hydro_{scenario}.h5"
        ),
        raw_list=(
            scenario_root
            / "lists"
            / "raw"
            / f"{tag}_raw_nopointlike_nside{nside}.txt"
        ),
        corrected_list=(
            scenario_root
            / "lists"
            / "corrected"
            / f"{tag}_rhocorr_nside{nside}.txt"
        ),
        raw_log=(
            scenario_root
            / "outputs"
            / "raw_clumpy"
            / run_tag
            / f"{basename}.halo_rendered.log"
        ),
        corrected_log=(
            scenario_root
            / "outputs"
            / "corrected_clumpy"
            / run_tag
            / f"{basename}.halo_rendered.log"
        ),
        total_fits=(
            scenario_root
            / "outputs"
            / "total"
            / run_tag
            / f"auriga_total_nside{nside}.fits"
        ),
    )


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ProductValidationError(f"{label} is missing: {path}")
    if path.stat().st_size == 0:
        raise ProductValidationError(f"{label} is empty: {path}")


def load_clumpy_list(path: Path) -> list[ListRow]:
    _require_nonempty_file(path, "CLUMPY list")
    rows: list[ListRow] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 13:
                raise ProductValidationError(
                    f"Malformed CLUMPY list row {path}:{line_number}: "
                    f"expected 13 fields, found {len(fields)}"
                )
            try:
                for index in (2, 3, 4, 5, 6, 7, 8, 10, 11, 12):
                    float(fields[index])
            except ValueError as error:
                raise ProductValidationError(
                    f"Malformed numeric value in {path}:{line_number}"
                ) from error
            rows.append(ListRow(fields[0], fields[1]))

    if not rows:
        raise ProductValidationError(f"CLUMPY list has no halo rows: {path}")
    names = [row.name for row in rows]
    if len(names) != len(set(names)):
        raise ProductValidationError(f"CLUMPY list contains duplicate NAME values: {path}")
    return rows


def load_rendered_log(path: Path) -> list[RenderedRow]:
    _require_nonempty_file(path, "rendered log")
    rows: list[RenderedRow] = []
    header_seen = False
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                candidate = tuple(stripped[1:].strip().split())
                if candidate == LOG_HEADER:
                    header_seen = True
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise ProductValidationError(
                    f"Malformed rendered-log row {path}:{line_number}: "
                    f"expected 5 fields, found {len(fields)}"
                )
            try:
                halo_index = int(fields[0])
                j_rendered = float(fields[3])
                int(fields[4])
            except ValueError as error:
                raise ProductValidationError(
                    f"Malformed rendered-log value in {path}:{line_number}"
                ) from error
            if not np.isfinite(j_rendered) or j_rendered < 0.0:
                raise ProductValidationError(
                    f"J_RENDERED must be finite and non-negative in "
                    f"{path}:{line_number}"
                )
            rows.append(
                RenderedRow(halo_index, fields[1], fields[2], j_rendered)
            )

    if not header_seen:
        raise ProductValidationError(
            f"Rendered log is missing the expected header: {path}"
        )
    if not rows:
        raise ProductValidationError(f"Rendered log has no halo rows: {path}")
    names = [row.name for row in rows]
    if len(names) != len(set(names)):
        raise ProductValidationError(
            f"Rendered log contains duplicate NAME values: {path}"
        )
    return rows


def validate_log_against_list(
    list_rows: Sequence[ListRow],
    log_rows: Sequence[RenderedRow],
    label: str,
) -> None:
    if len(log_rows) != len(list_rows):
        raise ProductValidationError(
            f"{label} log/list row-count mismatch: "
            f"log={len(log_rows)}, list={len(list_rows)}"
        )
    expected_indices = list(range(len(list_rows)))
    actual_indices = [row.halo_index for row in log_rows]
    if actual_indices != expected_indices:
        raise ProductValidationError(
            f"{label} HALO_INDEX is not exactly zero-based 0..N-1"
        )
    for index, (list_row, log_row) in enumerate(zip(list_rows, log_rows)):
        if log_row.name != list_row.name:
            raise ProductValidationError(
                f"{label} NAME mismatch at HALO_INDEX={index}: "
                f"list={list_row.name!r}, log={log_row.name!r}"
            )
        if log_row.halo_type != list_row.halo_type:
            raise ProductValidationError(
                f"{label} TYPE mismatch at HALO_INDEX={index}: "
                f"list={list_row.halo_type!r}, log={log_row.halo_type!r}"
            )


def validate_raw_corrected_sequence(
    raw_rows: Sequence[RenderedRow],
    corrected_rows: Sequence[RenderedRow],
) -> None:
    raw_sequence = [
        (row.halo_index, row.name, row.halo_type) for row in raw_rows
    ]
    corrected_sequence = [
        (row.halo_index, row.name, row.halo_type) for row in corrected_rows
    ]
    if raw_sequence != corrected_sequence:
        raise ProductValidationError(
            "Raw/corrected (HALO_INDEX, NAME, TYPE) sequences differ"
        )


def validate_total_fits(path: Path, nside: int) -> None:
    _require_nonempty_file(path, "total completion FITS")
    expected_npix = 12 * nside * nside
    expected = (
        ("JFACTOR", ["PIXEL", "Jtot", "Jsmooth", "Jextended", "Jpointlike"]),
        (
            "JFACTOR_PER_SR",
            [
                "PIXEL",
                "Jtot_per_sr",
                "Jsmooth_per_sr",
                "Jextended_per_sr",
                "Jpointlike_per_sr",
            ],
        ),
    )
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            lazy_load_hdus=False,
        ) as hdul:
            hdul.verify("exception")
            for name, required_columns in expected:
                matching_hdus = [hdu for hdu in hdul if hdu.name == name]
                if len(matching_hdus) != 1:
                    raise ProductValidationError(
                        f"Total FITS must contain exactly one {name} extension; "
                        f"found {len(matching_hdus)}"
                    )
                hdu = matching_hdus[0]
                if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    raise ProductValidationError(
                        f"Total FITS {name} extension is not a table"
                    )
                actual_columns = set(hdu.columns.names or [])
                missing_columns = [
                    column
                    for column in required_columns
                    if column not in actual_columns
                ]
                if missing_columns:
                    raise ProductValidationError(
                        f"Total FITS {name} is missing required column(s): "
                        f"{', '.join(missing_columns)}"
                    )
                if hdu.header.get("NSIDE") != nside:
                    raise ProductValidationError(
                        f"Total FITS {name} has NSIDE={hdu.header.get('NSIDE')!r}, "
                        f"expected {nside}"
                    )
                if hdu.header.get("NAXIS2") != expected_npix:
                    raise ProductValidationError(
                        f"Total FITS {name} has {hdu.header.get('NAXIS2')!r} rows, "
                        f"expected {expected_npix}"
                    )
                if hdu.data is None:
                    raise ProductValidationError(
                        f"Total FITS {name} table data are missing"
                    )
                if (
                    int(hdu.data["PIXEL"][0]) != 0
                    or int(hdu.data["PIXEL"][-1]) != expected_npix - 1
                ):
                    raise ProductValidationError(
                        f"Total FITS {name} PIXEL endpoints are invalid"
                    )
    except ProductValidationError:
        raise
    except (
        OSError,
        ValueError,
        IndexError,
        KeyError,
        TypeError,
        VerifyError,
    ) as error:
        raise ProductValidationError(
            f"Could not validate total completion FITS {path}: {error}"
        ) from error


def _decode_strings(values: Iterable[object]) -> list[str]:
    return [
        value.decode("utf-8")
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in values
    ]


def _validate_unit(
    column_name: str,
    raw_unit: str,
    expected_unit: u.UnitBase,
    path: Path,
) -> None:
    try:
        actual = u.Unit(raw_unit)
    except (TypeError, ValueError) as error:
        raise ProductValidationError(
            f"Invalid unit {raw_unit!r} for {column_name} in {path}"
        ) from error
    if actual != expected_unit:
        raise ProductValidationError(
            f"Wrong unit for {column_name} in {path}: "
            f"found {raw_unit!r}, expected {expected_unit}"
        )


def _catalog_layout(
    handle: h5py.File,
    path: Path,
) -> tuple[h5py.Dataset, h5py.Dataset, int, int]:
    """Validate catalogue structure/metadata without reading catalogue rows."""
    if "iteration_0" not in handle:
        raise ProductValidationError(f"Missing HDF5 group iteration_0 in {path}")
    group = handle["iteration_0"]
    if "data" not in group or "halo_name" not in group:
        raise ProductValidationError(
            f"Missing iteration_0/data or iteration_0/halo_name in {path}"
        )
    data = group["data"]
    names = group["halo_name"]
    if not isinstance(data, h5py.Dataset) or data.ndim != 2:
        raise ProductValidationError(
            f"iteration_0/data is not a two-dimensional dataset in {path}"
        )
    if not isinstance(names, h5py.Dataset) or names.ndim != 1:
        raise ProductValidationError(
            f"iteration_0/halo_name is not one-dimensional in {path}"
        )
    if names.shape[0] != data.shape[0]:
        raise ProductValidationError(
            f"HDF5 halo_name length mismatch in {path}: "
            f"names={names.shape[0]}, data={data.shape[0]}"
        )
    if "n_saved" not in group.attrs:
        raise ProductValidationError(f"Missing n_saved metadata in {path}")
    try:
        n_saved = int(group.attrs["n_saved"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ProductValidationError(f"Invalid n_saved metadata in {path}") from error
    if n_saved != data.shape[0]:
        raise ProductValidationError(
            f"HDF5 n_saved mismatch in {path}: "
            f"n_saved={n_saved}, data rows={data.shape[0]}"
        )
    if "column_names" not in data.attrs or "units" not in data.attrs:
        raise ProductValidationError(f"Missing column_names or units metadata in {path}")
    column_names = _decode_strings(data.attrs["column_names"])
    units = _decode_strings(data.attrs["units"])
    if len(column_names) != data.shape[1]:
        raise ProductValidationError(
            f"column_names length does not match HDF5 data width in {path}"
        )
    if len(units) != data.shape[1]:
        raise ProductValidationError(
            f"units length does not match HDF5 data width in {path}"
        )
    if len(column_names) != len(set(column_names)):
        raise ProductValidationError(f"Duplicate HDF5 column names in {path}")
    missing = [name for name in ("Js", "theta_s") if name not in column_names]
    if missing:
        raise ProductValidationError(
            f"Missing HDF5 column(s) in {path}: {', '.join(missing)}"
        )
    js_index = column_names.index("Js")
    theta_index = column_names.index("theta_s")
    _validate_unit("Js", units[js_index], u.GeV**2 / u.cm**5, path)
    _validate_unit("theta_s", units[theta_index], u.deg, path)
    return data, names, js_index, theta_index


def validate_catalog_metadata(path: Path) -> None:
    """Validate catalogue structure and units without scanning halo rows."""
    _require_nonempty_file(path, "HDF5 catalogue")
    try:
        with h5py.File(path, "r") as handle:
            _catalog_layout(handle, path)
    except ProductValidationError:
        raise
    except OSError as error:
        raise ProductValidationError(
            f"Could not open HDF5 catalogue {path}: {error}"
        ) from error


def load_catalog_values_indexed(
    path: Path,
    target_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return targeted catalog Js/theta_s in the requested NAME order."""
    _require_nonempty_file(path, "HDF5 catalogue")

    try:
        with h5py.File(path, "r") as handle:
            data, names, js_index, theta_index = _catalog_layout(handle, path)

            inferred_rows: list[int] = []
            row_to_name: dict[int, str] = {}
            for name in target_names:
                match = HALO_NAME_PATTERN.fullmatch(name)
                if match is None:
                    raise ProductValidationError(
                        f"Non-canonical halo NAME {name!r}; expected "
                        "hydro_res_<positive numeric suffix>"
                    )
                row_index = int(match.group(1)) - 1
                if row_index < 0 or row_index >= data.shape[0]:
                    raise ProductValidationError(
                        f"Halo NAME {name!r} implies HDF5 row {row_index}, "
                        f"outside valid range 0..{data.shape[0] - 1}"
                    )
                if row_index in row_to_name:
                    raise ProductValidationError(
                        f"Duplicate inferred HDF5 row {row_index} for halo NAMEs "
                        f"{row_to_name[row_index]!r} and {name!r}"
                    )
                row_to_name[row_index] = name
                inferred_rows.append(row_index)

            rows = np.asarray(inferred_rows, dtype=np.int64)
            sort_order = np.argsort(rows, kind="stable")
            sorted_rows = rows[sort_order]
            sorted_names = [target_names[index] for index in sort_order]
            stored_names = _decode_strings(names[sorted_rows])
            sorted_js = np.asarray(
                data[sorted_rows, js_index], dtype=np.float64
            )
            sorted_theta = np.asarray(
                data[sorted_rows, theta_index], dtype=np.float64
            )

            for sorted_position, (stored_name, expected_name) in enumerate(
                zip(stored_names, sorted_names)
            ):
                if stored_name != expected_name:
                    raise ProductValidationError(
                        f"HDF5 NAME mismatch at inferred row "
                        f"{int(sorted_rows[sorted_position])}: "
                        f"expected {expected_name!r}, found {stored_name!r}"
                    )
    except ProductValidationError:
        raise
    except OSError as error:
        raise ProductValidationError(f"Could not open HDF5 catalogue {path}: {error}") from error

    js = np.empty(len(sorted_js), dtype=np.float64)
    theta = np.empty(len(sorted_theta), dtype=np.float64)
    js[sort_order] = sorted_js
    theta[sort_order] = sorted_theta
    invalid_js = ~np.isfinite(js) | (js <= 0.0)
    invalid_theta = ~np.isfinite(theta) | (theta <= 0.0)
    if np.any(invalid_js):
        raise ProductValidationError(
            f"HDF5 contains {np.count_nonzero(invalid_js)} non-positive/non-finite "
            "matched Js value(s)"
        )
    if np.any(invalid_theta):
        raise ProductValidationError(
            f"HDF5 contains {np.count_nonzero(invalid_theta)} non-positive/non-finite "
            "matched theta_s value(s)"
        )
    return js, theta


def theta_pixel_deg(nside: int) -> float:
    theta_rad = math.sqrt(4.0 * math.pi / (12.0 * nside * nside))
    return math.degrees(theta_rad)


def validate_case(
    catalog_root: Path,
    clumpy_root: Path,
    scenario: str,
    repop_id: int,
    nside: int,
) -> tuple[ValidationRecord, CaseData | None]:
    paths = build_case_paths(
        catalog_root, clumpy_root, scenario, repop_id, nside
    )
    record = ValidationRecord(scenario=scenario, repop_id=repop_id)
    record.catalog_found = paths.catalog.is_file() and paths.catalog.stat().st_size > 0
    record.raw_list_found = paths.raw_list.is_file() and paths.raw_list.stat().st_size > 0
    record.corrected_list_found = (
        paths.corrected_list.is_file() and paths.corrected_list.stat().st_size > 0
    )
    record.raw_log_found = paths.raw_log.is_file() and paths.raw_log.stat().st_size > 0
    record.corrected_log_found = (
        paths.corrected_log.is_file() and paths.corrected_log.stat().st_size > 0
    )

    try:
        validate_catalog_metadata(paths.catalog)
        record.catalog_valid = True
    except ProductValidationError as error:
        record.reasons.append(str(error))

    try:
        validate_total_fits(paths.total_fits, nside)
        record.total_fits_valid = True
    except ProductValidationError as error:
        record.reasons.append(str(error))

    raw_list: list[ListRow] | None = None
    corrected_list: list[ListRow] | None = None
    raw_log: list[RenderedRow] | None = None
    corrected_log: list[RenderedRow] | None = None

    for attribute, loader, path, label in (
        ("raw_list", load_clumpy_list, paths.raw_list, "raw list"),
        ("corrected_list", load_clumpy_list, paths.corrected_list, "corrected list"),
        ("raw_log", load_rendered_log, paths.raw_log, "raw rendered log"),
        (
            "corrected_log",
            load_rendered_log,
            paths.corrected_log,
            "corrected rendered log",
        ),
    ):
        try:
            value = loader(path)
        except ProductValidationError as error:
            record.reasons.append(f"{label}: {error}")
            value = None
        if attribute == "raw_list":
            raw_list = value
        elif attribute == "corrected_list":
            corrected_list = value
        elif attribute == "raw_log":
            raw_log = value
        else:
            corrected_log = value

    if raw_list is not None and corrected_list is not None:
        raw_identity = [(row.name, row.halo_type) for row in raw_list]
        corrected_identity = [(row.name, row.halo_type) for row in corrected_list]
        if raw_identity != corrected_identity:
            record.reasons.append(
                "corrected list NAME/order/TYPE differs from raw list"
            )

    if raw_list is not None and raw_log is not None:
        try:
            validate_log_against_list(raw_list, raw_log, "raw")
            record.raw_list_matching_valid = True
        except ProductValidationError as error:
            record.reasons.append(str(error))

    if corrected_list is not None and corrected_log is not None:
        try:
            validate_log_against_list(corrected_list, corrected_log, "corrected")
            record.corrected_list_matching_valid = True
        except ProductValidationError as error:
            record.reasons.append(str(error))

    if raw_log is not None and corrected_log is not None:
        try:
            validate_raw_corrected_sequence(raw_log, corrected_log)
            record.raw_corrected_sequence_valid = True
        except ProductValidationError as error:
            record.reasons.append(str(error))

    j_catalog: np.ndarray | None = None
    theta_s: np.ndarray | None = None
    if raw_list is not None:
        try:
            j_catalog, theta_s = load_catalog_values_indexed(
                paths.catalog,
                [row.name for row in raw_list],
            )
            record.hdf5_matching_valid = True
            record.n_matched = len(raw_list)
        except ProductValidationError as error:
            record.reasons.append(str(error))
    elif not record.catalog_found:
        record.reasons.append(f"HDF5 catalogue is missing: {paths.catalog}")

    same_lists = (
        raw_list is not None
        and corrected_list is not None
        and [(row.name, row.halo_type) for row in raw_list]
        == [(row.name, row.halo_type) for row in corrected_list]
    )
    record.usable = bool(
        record.total_fits_valid
        and same_lists
        and record.raw_list_matching_valid
        and record.corrected_list_matching_valid
        and record.raw_corrected_sequence_valid
        and record.hdf5_matching_valid
    )
    if not record.usable:
        if not record.reasons:
            record.reasons.append("one or more required validation checks failed")
        return record, None

    assert raw_list is not None
    assert raw_log is not None
    assert corrected_log is not None
    assert j_catalog is not None
    assert theta_s is not None
    record.n_matched = len(raw_list)
    names = tuple(row.name for row in raw_list)
    j_raw = np.asarray([row.j_rendered for row in raw_log], dtype=np.float64)
    j_corr = np.asarray(
        [row.j_rendered for row in corrected_log], dtype=np.float64
    )
    r_raw = j_raw / j_catalog
    r_corr = j_corr / j_catalog
    delta_raw = 100.0 * np.abs(r_raw - 1.0)
    delta_corr = 100.0 * np.abs(r_corr - 1.0)
    x = theta_s / theta_pixel_deg(nside)
    if not all(
        np.all(np.isfinite(values))
        for values in (x, r_raw, r_corr, delta_raw, delta_corr)
    ):
        record.usable = False
        record.n_matched = 0
        record.reasons.append(
            "derived x, R, or delta values contain non-finite values"
        )
        return record, None
    case = CaseData(
        scenario=scenario,
        repop_id=repop_id,
        names=names,
        j_catalog=j_catalog,
        theta_s_deg=theta_s,
        j_rendered_raw=j_raw,
        j_rendered_corr=j_corr,
        x=x,
        r_raw=r_raw,
        r_corr=r_corr,
        delta_raw=delta_raw,
        delta_corr=delta_corr,
    )
    return record, case


def per_repopulation_summary(case: CaseData) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": case.scenario,
        "repop_id": case.repop_id,
        "n_matched": len(case.names),
    }
    for stage, ratios, deltas in (
        ("raw", case.r_raw, case.delta_raw),
        ("corr", case.r_corr, case.delta_corr),
    ):
        row[f"r_{stage}_median"] = float(np.percentile(ratios, 50.0))
        row[f"r_{stage}_p16"] = float(np.percentile(ratios, 16.0))
        row[f"r_{stage}_p84"] = float(np.percentile(ratios, 84.0))
        for percentile in (68, 95, 99):
            row[f"delta_{stage}_p{percentile}"] = float(
                np.percentile(deltas, percentile)
            )
        for threshold in (1, 5, 10):
            row[f"frac_delta_{stage}_lt_{threshold}pct"] = float(
                np.mean(deltas < float(threshold))
            )
    return row


def matched_halo_diagnostics_rows(
    cases: Sequence[CaseData],
    nside: int,
) -> list[dict[str, object]]:
    theta_p_deg = theta_pixel_deg(nside)
    rows: list[dict[str, object]] = []
    for case in cases:
        for index, halo_name in enumerate(case.names):
            rows.append(
                {
                    "scenario": case.scenario,
                    "repop_id": case.repop_id,
                    "halo_name": halo_name,
                    "theta_s_deg": float(case.theta_s_deg[index]),
                    "theta_p_deg": theta_p_deg,
                    "theta_s_over_theta_p": float(case.x[index]),
                    "J_catalog": float(case.j_catalog[index]),
                    "J_raw": float(case.j_rendered_raw[index]),
                    "R_raw": float(case.r_raw[index]),
                    "delta_raw_percent": float(case.delta_raw[index]),
                    "J_corr": float(case.j_rendered_corr[index]),
                    "R_corr": float(case.r_corr[index]),
                    "delta_corr_percent": float(case.delta_corr[index]),
                }
            )
    return rows


def ensemble_summary_rows(
    per_repopulation: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        selected = [row for row in per_repopulation if row["scenario"] == scenario]
        if not selected:
            continue
        total_matched = sum(int(row["n_matched"]) for row in selected)
        statistics = [
            key for key in selected[0] if key not in {"scenario", "repop_id"}
        ]
        for statistic in statistics:
            values = np.asarray(
                [float(row[statistic]) for row in selected], dtype=np.float64
            )
            rows.append(
                {
                    "scenario": scenario,
                    "statistic": statistic,
                    "ensemble_median": float(np.percentile(values, 50.0)),
                    "ensemble_p16": float(np.percentile(values, 16.0)),
                    "ensemble_p84": float(np.percentile(values, 84.0)),
                    "n_repopulations": len(selected),
                    "total_matched_halos": total_matched,
                }
            )
    return rows


def common_log_edges(cases: Sequence[CaseData], n_bins: int) -> np.ndarray:
    minima = [float(np.min(case.x)) for case in cases]
    maxima = [float(np.max(case.x)) for case in cases]
    lower = min(minima)
    upper = max(maxima)
    if not np.isfinite(lower) or lower <= 0.0 or not np.isfinite(upper):
        raise ValueError("theta_s/theta_p must be finite and positive.")
    if lower == upper:
        lower /= math.sqrt(10.0)
        upper *= math.sqrt(10.0)
    return np.geomspace(lower, upper, n_bins + 1)


def binned_medians(
    x: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    result = np.full(len(edges) - 1, np.nan, dtype=np.float64)
    indices = np.searchsorted(edges, x, side="right") - 1
    indices[x == edges[-1]] = len(edges) - 2
    for bin_index in range(len(result)):
        selected = values[indices == bin_index]
        if selected.size:
            result[bin_index] = float(np.median(selected))
    return result


def pointwise_percentiles(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional ensemble array.")
    n_columns = values.shape[1]
    lower = np.full(n_columns, np.nan)
    median = np.full(n_columns, np.nan)
    upper = np.full(n_columns, np.nan)
    count = np.zeros(n_columns, dtype=np.int64)
    for column in range(n_columns):
        finite = values[:, column][np.isfinite(values[:, column])]
        count[column] = finite.size
        if finite.size:
            lower[column], median[column], upper[column] = np.percentile(
                finite, [16.0, 50.0, 84.0]
            )
    return lower, median, upper, count


def build_trend_summaries(
    cases: Sequence[CaseData],
    edges: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    per_case_curves: dict[tuple[str, str], np.ndarray] = {}
    centers = np.sqrt(edges[:-1] * edges[1:])
    for scenario in SCENARIOS:
        selected = [case for case in cases if case.scenario == scenario]
        if not selected:
            continue
        for stage in ("raw", "corr"):
            curves = np.asarray(
                [
                    binned_medians(
                        case.x,
                        case.r_raw if stage == "raw" else case.r_corr,
                        edges,
                    )
                    for case in selected
                ]
            )
            per_case_curves[(scenario, stage)] = curves
            lower, median, upper, count = pointwise_percentiles(curves)
            for index, center in enumerate(centers):
                rows.append(
                    {
                        "scenario": scenario,
                        "stage": stage,
                        "x_left": edges[index],
                        "x_right": edges[index + 1],
                        "x_center": center,
                        "ensemble_median": median[index],
                        "ensemble_p16": lower[index],
                        "ensemble_p84": upper[index],
                        "n_contributing_repopulations": int(count[index]),
                    }
                )
    return rows, per_case_curves


def common_cdf_grid(cases: Sequence[CaseData], n_points: int) -> np.ndarray:
    arrays = [
        values
        for case in cases
        for values in (case.delta_raw, case.delta_corr)
    ]
    maximum = max(float(np.max(values)) for values in arrays)
    positive_minima = [
        float(np.min(values[values > 0.0]))
        for values in arrays
        if np.any(values > 0.0)
    ]
    if maximum == 0.0:
        return np.linspace(0.0, 1.0, n_points)
    lower = min(positive_minima)
    return np.concatenate(
        ([0.0], np.geomspace(lower, maximum, n_points - 1))
    )


def empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if sorted_values.size == 0:
        raise ValueError("Cannot construct an empirical CDF from no values.")
    return np.searchsorted(sorted_values, grid, side="right") / sorted_values.size


def build_cdf_summaries(
    cases: Sequence[CaseData],
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    cdfs_by_group: dict[tuple[str, str], np.ndarray] = {}
    for scenario in SCENARIOS:
        selected = [case for case in cases if case.scenario == scenario]
        if not selected:
            continue
        for stage in ("raw", "corr"):
            cdfs = np.asarray(
                [
                    empirical_cdf(
                        case.delta_raw if stage == "raw" else case.delta_corr,
                        grid,
                    )
                    for case in selected
                ]
            )
            cdfs_by_group[(scenario, stage)] = cdfs
            lower, median, upper, count = pointwise_percentiles(cdfs)
            for index, delta in enumerate(grid):
                rows.append(
                    {
                        "scenario": scenario,
                        "stage": stage,
                        "delta_percent": delta,
                        "ensemble_median": median[index],
                        "ensemble_p16": lower[index],
                        "ensemble_p84": upper[index],
                        "n_contributing_repopulations": int(count[index]),
                    }
                )
    return rows, cdfs_by_group


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    trend_rows: Sequence[dict[str, object]],
    cdf_rows: Sequence[dict[str, object]],
    output_dir: Path,
    nside: int,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.8),
        constrained_layout=True,
    )
    trend_axes = {"raw": axes[0, 0], "corr": axes[0, 1]}
    cdf_axes = {"raw": axes[1, 0], "corr": axes[1, 1]}

    for stage, axis in trend_axes.items():
        for scenario in SCENARIOS:
            selected = [
                row
                for row in trend_rows
                if row["scenario"] == scenario
                and row["stage"] == stage
                and int(row["n_contributing_repopulations"]) > 0
            ]
            if not selected:
                continue
            x = np.asarray([float(row["x_center"]) for row in selected])
            median = np.asarray(
                [float(row["ensemble_median"]) for row in selected]
            )
            lower = np.asarray([float(row["ensemble_p16"]) for row in selected])
            upper = np.asarray([float(row["ensemble_p84"]) for row in selected])
            color = SCENARIO_COLORS[scenario]
            axis.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
            axis.plot(
                x,
                median,
                color=color,
                linewidth=1.9,
                label=scenario.capitalize(),
            )
        axis.axhline(1.0, color="0.45", linestyle="--", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_xlabel(r"$\theta_s/\theta_p$")
        axis.set_ylabel(
            r"$R_{\rm raw}=J_{\rm rendered}/J_{\rm catalog}$"
            if stage == "raw"
            else r"$R_{\rm corr}=J_{\rm rendered}/J_{\rm catalog}$"
        )
        axis.set_title("Raw rendering" if stage == "raw" else "Corrected rendering")
        axis.grid(False, which="both")
        axis.tick_params(direction="in", which="both", top=True, right=True)
        axis.legend(frameon=False)

    for stage, axis in cdf_axes.items():
        maximum = 0.0
        minimum_positive: float | None = None
        for scenario in SCENARIOS:
            selected = [
                row
                for row in cdf_rows
                if row["scenario"] == scenario and row["stage"] == stage
            ]
            if not selected:
                continue
            x = np.asarray([float(row["delta_percent"]) for row in selected])
            median = np.asarray(
                [float(row["ensemble_median"]) for row in selected]
            )
            lower = np.asarray([float(row["ensemble_p16"]) for row in selected])
            upper = np.asarray([float(row["ensemble_p84"]) for row in selected])
            positive = x[x > 0.0]
            maximum = max(maximum, float(np.max(x)))
            if positive.size:
                candidate = float(np.min(positive))
                minimum_positive = (
                    candidate
                    if minimum_positive is None
                    else min(minimum_positive, candidate)
                )
            color = SCENARIO_COLORS[scenario]
            axis.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
            axis.plot(
                x,
                median,
                color=color,
                linewidth=1.9,
                label=scenario.capitalize(),
            )
        if maximum > 0.0 and minimum_positive is not None:
            axis.set_xscale("symlog", linthresh=minimum_positive)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel(
            r"$\delta_{\rm raw}=100\,|R_{\rm raw}-1|\;[\%]$"
            if stage == "raw"
            else r"$\delta_{\rm corr}=100\,|R_{\rm corr}-1|\;[\%]$"
        )
        axis.set_ylabel("Cumulative fraction of extended halos")
        axis.set_title(
            "Raw residual CDF" if stage == "raw" else "Corrected residual CDF"
        )
        axis.grid(False, which="both")
        axis.tick_params(direction="in", which="both", top=True, right=True)
        axis.legend(frameon=False, loc="lower right")

    figure.suptitle(rf"CLUMPY rendering correction diagnostic, NSIDE={nside}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"clumpy_rendering_corrections_nside{nside}"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _status(value: bool) -> str:
    return "yes" if value else "no"


def print_validation_record(record: ValidationRecord) -> None:
    fields = (
        f"scenario={record.scenario}",
        f"repop_id={record.repop_id}",
        f"catalog_found={_status(record.catalog_found)}",
        f"catalog_valid={_status(record.catalog_valid)}",
        f"raw_list_found={_status(record.raw_list_found)}",
        f"corrected_list_found={_status(record.corrected_list_found)}",
        f"raw_log_found={_status(record.raw_log_found)}",
        f"corrected_log_found={_status(record.corrected_log_found)}",
        f"total_fits_valid={_status(record.total_fits_valid)}",
        f"raw_list_matching={_status(record.raw_list_matching_valid)}",
        f"corrected_list_matching={_status(record.corrected_list_matching_valid)}",
        f"raw_corrected_sequence={_status(record.raw_corrected_sequence_valid)}",
        f"hdf5_matching={_status(record.hdf5_matching_valid)}",
        f"n_matched={record.n_matched}",
        f"usable={_status(record.usable)}",
    )
    print(" | ".join(fields))
    if not record.usable:
        print(f"  exclusion_reason={record.exclusion_reason}")


def print_ensemble_summary(rows: Sequence[dict[str, object]]) -> None:
    print("\nEqual-weight ensemble summaries")
    for row in rows:
        print(
            f"{row['scenario']:9s} {row['statistic']:30s} "
            f"median={float(row['ensemble_median']):.8g} "
            f"p16={float(row['ensemble_p16']):.8g} "
            f"p84={float(row['ensemble_p84']):.8g}"
        )
    for scenario in SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        if selected:
            print(
                f"{scenario:9s} n_repops={selected[0]['n_repopulations']} "
                f"total_matched_halos={selected[0]['total_matched_halos']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        validate_cli_args(args)
        repop_ids = parse_repop_ids(args)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    records: list[ValidationRecord] = []
    cases: list[CaseData] = []
    for scenario in scenarios_requested(args.scenario):
        for repop_id in repop_ids:
            record, case = validate_case(
                catalog_root=args.catalog_root,
                clumpy_root=args.clumpy_root,
                scenario=scenario,
                repop_id=repop_id,
                nside=args.nside,
            )
            records.append(record)
            print_validation_record(record)
            if case is not None:
                cases.append(case)

    if args.validate_only:
        return 0 if all(record.usable for record in records) else 1

    if not cases:
        print("ERROR: no requested repopulations passed validation.", file=sys.stderr)
        return 1

    per_repop = [per_repopulation_summary(case) for case in cases]
    matched_halos = matched_halo_diagnostics_rows(cases, args.nside)
    ensemble = ensemble_summary_rows(per_repop)
    edges = common_log_edges(cases, args.x_bins)
    trend_rows, _ = build_trend_summaries(cases, edges)
    cdf_grid = common_cdf_grid(cases, args.cdf_points)
    cdf_rows, _ = build_cdf_summaries(cases, cdf_grid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched_halos_path = args.output_dir / "matched_halo_diagnostics.csv"
    per_repop_path = args.output_dir / "per_repopulation_summary.csv"
    ensemble_path = args.output_dir / "ensemble_summary.csv"
    trend_path = args.output_dir / "binned_trend_summary.csv"
    cdf_path = args.output_dir / "cdf_summary.csv"
    write_csv(matched_halos_path, matched_halos)
    write_csv(per_repop_path, per_repop)
    write_csv(ensemble_path, ensemble)
    write_csv(trend_path, trend_rows)
    write_csv(cdf_path, cdf_rows)
    try:
        png, pdf = make_figure(trend_rows, cdf_rows, args.output_dir, args.nside)
    except Exception as error:
        print(f"ERROR: failed to create diagnostic figure: {error}", file=sys.stderr)
        return 1

    print_ensemble_summary(ensemble)
    print(f"\nSaved: {matched_halos_path}")
    print(f"\nSaved: {per_repop_path}")
    print(f"Saved: {ensemble_path}")
    print(f"Saved: {trend_path}")
    print(f"Saved: {cdf_path}")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    excluded = sum(not record.usable for record in records)
    if excluded:
        print(f"Excluded requested repopulations: {excluded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
