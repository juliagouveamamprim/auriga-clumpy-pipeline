#!/usr/bin/env python3
"""Consolidate halo-level raw/corrected CLUMPY correction diagnostics.

The full-repopulation HDF5 catalogues are accessed sparsely.  Halo rows are
inferred from the canonical ``hydro_res_<N>`` names, sorted for h5py fancy
indexing, and only ``Js``, ``theta_s``, and the corresponding stored names are
read.  No full catalogue dataset is materialized.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import correct_rhos_from_clumpy_raw as correction  # noqa: E402
import analyze_clumpy_rendering_corrections as analysis  # noqa: E402


DEFAULT_NSIDE = 2048
DEFAULT_START_ID = 0
DEFAULT_END_ID = 21
DEFAULT_CATALOG_ROOT = REPOSITORY_ROOT / "outputs"
DEFAULT_CLUMPY_ROOT = REPOSITORY_ROOT / "outputs" / "clumpy"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "diagnostics" / "corrections" / "data"
RHO_RATIO_RTOL = 5.0e-8
RATIO_RTOL = 1.0e-12

FINAL_COLUMNS = [
    "scenario",
    "repop_id",
    "halo_index",
    "halo_name",
    "theta_s_deg",
    "J_s",
    "J_raw",
    "R_raw",
    "rho_s_raw",
    "rho_s_corr",
    "rho_corr_over_raw",
    "J_corr",
    "R_corr",
    "npix_raw",
    "npix_corr",
]


class ConsolidationError(RuntimeError):
    """A required correction product is missing or inconsistent."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate halo-level raw/corrected CLUMPY correction "
            "diagnostics over an inclusive repopulation range."
        )
    )
    parser.add_argument("scenario", choices=("fragile", "resilient"))
    parser.add_argument("--start-id", type=int, default=DEFAULT_START_ID)
    parser.add_argument(
        "--end-id",
        type=int,
        default=DEFAULT_END_ID,
        help="Inclusive final repopulation ID (default: 21).",
    )
    parser.add_argument("--nside", type=int, default=DEFAULT_NSIDE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--clumpy-root", type=Path, default=DEFAULT_CLUMPY_ROOT)
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.start_id < 0 or args.end_id < 0:
        raise ValueError("Repopulation IDs must be non-negative.")
    if args.start_id > args.end_id:
        raise ValueError("Require --start-id <= --end-id (inclusive range).")
    if args.nside <= 0 or (args.nside & (args.nside - 1)) != 0:
        raise ValueError("--nside must be a positive power of two.")


def default_output_path(scenario: str, start_id: int, end_id: int) -> Path:
    return DEFAULT_OUTPUT_DIR / (
        f"corrections_{scenario}_repop{start_id:04d}_{end_id:04d}.csv"
    )


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ConsolidationError(f"{label} is missing or not a regular file: {path}")
    if path.stat().st_size == 0:
        raise ConsolidationError(f"{label} is empty: {path}")


def _read_sparse_hdf5_rows(
    data: h5py.Dataset,
    stored_names: h5py.Dataset,
    sorted_rows: np.ndarray,
    js_index: int,
    theta_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only selected names and the selected Js/theta_s cells."""
    names = np.asarray(stored_names[sorted_rows])
    js = np.asarray(data[sorted_rows, js_index], dtype=np.float64)
    theta = np.asarray(data[sorted_rows, theta_index], dtype=np.float64)
    return names, js, theta


def load_catalog_values_sparse(
    catalog_path: Path,
    target_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load HDF5 row indices, Js, and theta_s with one sparse file access."""
    catalog_path = Path(catalog_path)
    _require_nonempty_file(catalog_path, "HDF5 catalogue")
    if not target_names:
        raise ConsolidationError("Requested HDF5 halo-name sequence is empty")
    if len(target_names) != len(set(target_names)):
        raise ConsolidationError("Requested HDF5 halo names are not unique")

    try:
        with h5py.File(catalog_path, "r") as handle:
            data, stored_names, js_index, theta_index = analysis._catalog_layout(
                handle, catalog_path
            )
            inferred_rows: list[int] = []
            row_to_name: dict[int, str] = {}
            for name in target_names:
                match = analysis.HALO_NAME_PATTERN.fullmatch(name)
                if match is None:
                    raise ConsolidationError(
                        f"Non-canonical halo name {name!r}; expected "
                        "hydro_res_<positive numeric suffix>"
                    )
                row_index = int(match.group(1)) - 1
                if row_index < 0 or row_index >= data.shape[0]:
                    raise ConsolidationError(
                        f"Halo name {name!r} implies HDF5 row {row_index}, "
                        f"outside 0..{data.shape[0] - 1}"
                    )
                if row_index in row_to_name:
                    raise ConsolidationError(
                        f"HDF5 row {row_index} is implied by both "
                        f"{row_to_name[row_index]!r} and {name!r}"
                    )
                row_to_name[row_index] = name
                inferred_rows.append(row_index)

            rows = np.asarray(inferred_rows, dtype=np.int64)
            sort_order = np.argsort(rows, kind="stable")
            sorted_rows = rows[sort_order]
            selected_names, sorted_js, sorted_theta = _read_sparse_hdf5_rows(
                data,
                stored_names,
                sorted_rows,
                js_index,
                theta_index,
            )
            decoded_names = analysis._decode_strings(selected_names)
            expected_names = [target_names[index] for index in sort_order]
            for row_index, expected_name, actual_name in zip(
                sorted_rows, expected_names, decoded_names
            ):
                if actual_name != expected_name:
                    raise ConsolidationError(
                        f"HDF5 identity mismatch at row {int(row_index)}: "
                        f"expected {expected_name!r}, found {actual_name!r}"
                    )
    except ConsolidationError:
        raise
    except analysis.ProductValidationError as error:
        raise ConsolidationError(str(error)) from error
    except OSError as error:
        raise ConsolidationError(
            f"Could not open HDF5 catalogue {catalog_path}: {error}"
        ) from error

    js = np.empty(len(sorted_js), dtype=np.float64)
    theta = np.empty(len(sorted_theta), dtype=np.float64)
    js[sort_order] = sorted_js
    theta[sort_order] = sorted_theta
    invalid_js = ~np.isfinite(js) | (js <= 0.0)
    invalid_theta = ~np.isfinite(theta) | (theta <= 0.0)
    if np.any(invalid_js):
        raise ConsolidationError(
            f"J_s contains {np.count_nonzero(invalid_js)} non-finite/non-positive value(s)"
        )
    if np.any(invalid_theta):
        raise ConsolidationError(
            "theta_s_deg contains "
            f"{np.count_nonzero(invalid_theta)} non-finite/non-positive value(s)"
        )
    return rows, js, theta


def _validate_same_lists(raw_list: pd.DataFrame, corr_list: pd.DataFrame) -> None:
    if len(raw_list) != len(corr_list):
        raise ConsolidationError(
            f"Raw/corrected list row-count mismatch: {len(raw_list)} != {len(corr_list)}"
        )

    exact_columns = ["halo_name", "type", "prof", "p1", "p2", "p3"]
    numeric_columns = [
        "l_deg",
        "b_deg",
        "d_kpc",
        "z",
        "Rdelta_kpc",
        "rs_kpc",
    ]
    for column in exact_columns:
        if not raw_list[column].equals(corr_list[column]):
            raise ConsolidationError(
                f"Raw/corrected list field {column!r} differs; only rhos may change"
            )
    for column in numeric_columns:
        raw_values = raw_list[column].to_numpy(dtype=np.float64)
        corr_values = corr_list[column].to_numpy(dtype=np.float64)
        if not np.array_equal(raw_values, corr_values):
            raise ConsolidationError(
                f"Raw/corrected list field {column!r} differs; only rhos may change"
            )


def _validate_raw_corrected_sequence(
    raw_log: pd.DataFrame,
    corr_log: pd.DataFrame,
) -> None:
    raw_sequence = list(
        zip(
            raw_log["HALO_INDEX"].tolist(),
            raw_log["halo_name"].tolist(),
            raw_log["rendered_type"].tolist(),
        )
    )
    corr_sequence = list(
        zip(
            corr_log["HALO_INDEX"].tolist(),
            corr_log["halo_name"].tolist(),
            corr_log["rendered_type"].tolist(),
        )
    )
    if raw_sequence != corr_sequence:
        raise ConsolidationError(
            "Raw/corrected (HALO_INDEX, NAME, TYPE) sequences differ"
        )


def _require_finite_positive(values: np.ndarray, label: str) -> None:
    invalid = ~np.isfinite(values) | (values <= 0.0)
    if np.any(invalid):
        raise ConsolidationError(
            f"{label} contains {np.count_nonzero(invalid)} "
            "non-finite/non-positive value(s)"
        )


def _validate_pixels(j_values: np.ndarray, pixels: np.ndarray, label: str) -> None:
    if np.any(pixels < 0):
        raise ConsolidationError(f"{label} contains negative values")
    invalid_positive_j = (j_values > 0.0) & (pixels <= 0)
    if np.any(invalid_positive_j):
        raise ConsolidationError(
            f"{label} must be positive whenever the corresponding J is positive"
        )


def build_repopulation_table(
    catalog_root: Path,
    clumpy_root: Path,
    scenario: str,
    repop_id: int,
    nside: int,
) -> pd.DataFrame:
    paths = analysis.build_case_paths(
        Path(catalog_root), Path(clumpy_root), scenario, repop_id, nside
    )
    required = (
        (paths.catalog, "HDF5 catalogue"),
        (paths.raw_list, "raw CLUMPY list"),
        (paths.corrected_list, "corrected CLUMPY list"),
        (paths.raw_log, "raw rendered log"),
        (paths.corrected_log, "corrected rendered log"),
    )
    for path, label in required:
        _require_nonempty_file(path, label)

    raw_list = correction.load_clumpy_list(paths.raw_list)
    corr_list = correction.load_clumpy_list(paths.corrected_list)
    raw_log = correction.load_rendered_log(paths.raw_log)
    corr_log = correction.load_rendered_log(paths.corrected_log)

    counts = {
        "raw_list": len(raw_list),
        "corrected_list": len(corr_list),
        "raw_log": len(raw_log),
        "corrected_log": len(corr_log),
    }
    if len(set(counts.values())) != 1:
        raise ConsolidationError(
            "Raw/corrected list/log row-count mismatch: "
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )

    try:
        correction.validate_raw_correspondence(raw_list, raw_log)
        correction.validate_raw_correspondence(corr_list, corr_log)
    except RuntimeError as error:
        raise ConsolidationError(str(error)) from error
    _validate_same_lists(raw_list, corr_list)
    _validate_raw_corrected_sequence(raw_log, corr_log)

    names = raw_list["halo_name"].tolist()
    hdf5_rows, j_s, theta_s = load_catalog_values_sparse(paths.catalog, names)
    try:
        raw_matched = correction.build_matched_table(raw_list, raw_log, hdf5_rows, j_s)
    except RuntimeError as error:
        raise ConsolidationError(str(error)) from error

    j_raw = raw_matched["J_raw"].to_numpy(dtype=np.float64)
    r_raw = raw_matched["R_raw"].to_numpy(dtype=np.float64)
    j_corr = corr_log["J_raw"].to_numpy(dtype=np.float64)
    rho_s_raw = 4.0 * raw_list["rhos_raw"].to_numpy(dtype=np.float64)
    rho_s_corr = 4.0 * corr_list["rhos_raw"].to_numpy(dtype=np.float64)
    rho_ratio = rho_s_corr / rho_s_raw
    r_corr = j_corr / j_s
    npix_raw = raw_log["npix_touched"].to_numpy(dtype=np.int64)
    npix_corr = corr_log["npix_touched"].to_numpy(dtype=np.int64)

    for values, label in (
        (j_s, "J_s"),
        (j_raw, "J_raw"),
        (j_corr, "J_corr"),
        (rho_s_raw, "rho_s_raw"),
        (rho_s_corr, "rho_s_corr"),
        (r_raw, "R_raw"),
        (r_corr, "R_corr"),
        (rho_ratio, "rho_corr_over_raw"),
    ):
        _require_finite_positive(values, label)
    _validate_pixels(j_raw, npix_raw, "npix_raw")
    _validate_pixels(j_corr, npix_corr, "npix_corr")

    expected_r_raw = j_raw / j_s
    if not np.allclose(r_raw, expected_r_raw, rtol=RATIO_RTOL, atol=0.0):
        raise ConsolidationError("R_raw is inconsistent with J_raw / J_s")
    expected_r_corr = j_corr / j_s
    if not np.allclose(r_corr, expected_r_corr, rtol=RATIO_RTOL, atol=0.0):
        raise ConsolidationError("R_corr is inconsistent with J_corr / J_s")
    expected_rho_ratio = 1.0 / np.sqrt(r_raw)
    if not np.allclose(
        rho_ratio,
        expected_rho_ratio,
        rtol=RHO_RATIO_RTOL,
        atol=0.0,
    ):
        max_relative = float(np.max(np.abs(rho_ratio / expected_rho_ratio - 1.0)))
        raise ConsolidationError(
            "rho_corr_over_raw is inconsistent with 1/sqrt(R_raw): "
            f"maximum relative difference={max_relative:.6e}, "
            f"rtol={RHO_RATIO_RTOL:.1e}"
        )

    table = pd.DataFrame(
        {
            "scenario": scenario,
            "repop_id": np.full(len(raw_list), repop_id, dtype=np.int64),
            "halo_index": raw_log["HALO_INDEX"].to_numpy(dtype=np.int64),
            "halo_name": names,
            "theta_s_deg": theta_s,
            "J_s": j_s,
            "J_raw": j_raw,
            "R_raw": r_raw,
            "rho_s_raw": rho_s_raw,
            "rho_s_corr": rho_s_corr,
            "rho_corr_over_raw": rho_ratio,
            "J_corr": j_corr,
            "R_corr": r_corr,
            "npix_raw": npix_raw,
            "npix_corr": npix_corr,
        },
        columns=FINAL_COLUMNS,
    )
    return table


def validate_consolidated_table(
    table: pd.DataFrame,
    scenario: str,
    start_id: int,
    end_id: int,
) -> None:
    if list(table.columns) != FINAL_COLUMNS:
        raise ConsolidationError(
            f"Final schema mismatch: expected {FINAL_COLUMNS}, found {list(table.columns)}"
        )
    if table.empty:
        raise ConsolidationError("Consolidated table is empty")

    expected_ids = set(range(start_id, end_id + 1))
    actual_ids = set(table["repop_id"].astype(int).tolist())
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ConsolidationError(
            f"Global repopulation coverage mismatch: missing={missing}, extra={extra}"
        )
    scenarios = set(table["scenario"].tolist())
    if scenarios != {scenario}:
        raise ConsolidationError(
            f"Scenario mixing detected: expected {scenario!r}, found {sorted(scenarios)}"
        )
    for key in (
        ["scenario", "repop_id", "halo_index"],
        ["scenario", "repop_id", "halo_name"],
    ):
        if table.duplicated(key).any():
            raise ConsolidationError(f"Duplicate final key detected: {tuple(key)}")

    for repop_id in range(start_id, end_id + 1):
        selected = table.loc[table["repop_id"] == repop_id]
        if selected.empty:
            raise ConsolidationError(f"repop_{repop_id:04d} has zero halos")
        indices = selected["halo_index"].to_numpy(dtype=np.int64)
        expected = np.arange(len(selected), dtype=np.int64)
        if not np.array_equal(indices, expected):
            raise ConsolidationError(
                f"repop_{repop_id:04d} halo_index is not contiguous 0..N-1"
            )


def consolidate(
    catalog_root: Path,
    clumpy_root: Path,
    scenario: str,
    start_id: int,
    end_id: int,
    nside: int,
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for repop_id in range(start_id, end_id + 1):
        try:
            table = build_repopulation_table(
                catalog_root,
                clumpy_root,
                scenario,
                repop_id,
                nside,
            )
        except (
            ConsolidationError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as error:
            raise ConsolidationError(
                f"scenario={scenario}, repop_{repop_id:04d}: {error}"
            ) from error
        tables.append(table)
        print(f"repop_{repop_id:04d}: {len(table)} halos")

    combined = pd.concat(tables, ignore_index=True)
    combined = combined.sort_values(
        ["repop_id", "halo_index"], kind="stable", ignore_index=True
    )
    validate_consolidated_table(combined, scenario, start_id, end_id)
    return combined


def write_csv_atomic(table: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            table.to_csv(stream, index=False, float_format="%.17g")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        validate_cli_args(args)
        output_path = args.output or default_output_path(
            args.scenario, args.start_id, args.end_id
        )
        table = consolidate(
            catalog_root=args.catalog_root,
            clumpy_root=args.clumpy_root,
            scenario=args.scenario,
            start_id=args.start_id,
            end_id=args.end_id,
            nside=args.nside,
        )
        write_csv_atomic(table, output_path)
    except (ConsolidationError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Total halos ({args.scenario}): {len(table)}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
