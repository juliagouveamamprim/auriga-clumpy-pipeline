#!/usr/bin/env python3
"""Count pre-brightness-cut dark subhalos by angular classification."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Iterable

import h5py
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.prepare_subhalo_components import (  # noqa: E402
    CHUNK_SIZE,
    ITERATION,
    ROUND_UP_DECIMALS,
    build_valid_mask,
    healpix_pixel_size_deg,
    round_up,
)


SCHEMA_VERSION = 1
EXPECTED_REPOPULATIONS = 500
VALID_SCENARIOS = ("fragile", "resilient")
NSIDE = 2048
MAX_VMAX_KM_S = 8.0
THETA_CUT_DEG = 0.03

REQUIRED_COLUMNS = (
    "Js",
    "D_Earth",
    "Vmax",
    "theta_s",
    "r_s",
    "rho_s",
    "Xearth",
    "Yearth",
    "Zearth",
)

OUTPUT_COLUMNS = (
    "schema_version",
    "scenario",
    "repop_id",
    "source_h5",
    "iteration",
    "n_input_rows",
    "n_physical_valid",
    "n_vmax_excluded",
    "n_total",
    "n_pointlike",
    "n_extended",
    "pointlike_fraction",
    "extended_fraction",
    "max_vmax_km_s",
    "vmax_selection",
    "nside",
    "theta_pix_deg",
    "theta_cut_deg",
    "round_up_decimals",
    "chunk_size",
)

SUMMARY_COLUMNS = (
    "scenario",
    "n_repopulations",
    "repop_start",
    "repop_end",
    "mean_n_total",
    "mean_n_pointlike",
    "mean_n_extended",
    "mean_pointlike_fraction",
    "mean_extended_fraction",
    "aggregate_pointlike_fraction",
    "aggregate_extended_fraction",
    "max_vmax_km_s",
    "vmax_selection",
    "nside",
    "theta_cut_deg",
)

FIGURE9_REQUIRED_COLUMNS = {
    "scenario",
    "repop_id",
    "n_valid_total",
    "n_pointlike_total",
    "n_extended_total",
    "max_vmax_km_s",
    "theta_min_deg",
}


def parse_scenarios(text: str) -> tuple[str, ...]:
    scenarios = tuple(
        dict.fromkeys(part.strip().lower() for part in text.split(",") if part.strip())
    )
    invalid = sorted(set(scenarios) - set(VALID_SCENARIOS))
    if invalid:
        raise argparse.ArgumentTypeError(
            "Invalid scenario(s): " + ", ".join(invalid)
        )
    if not scenarios:
        raise argparse.ArgumentTypeError("At least one scenario is required.")
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count production-valid Vmax <= 8 km/s subhalos before numerical "
            "brightness cuts, split at theta_s = 0.03 deg. Existing validated "
            "rows are resumed automatically."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Root containing repop_XXXX/fullrepop_hydro_<scenario>.h5 files."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Atomically checkpointed per-repopulation CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=(
            "Scenario summary CSV. Defaults to <output stem>_summary.csv."
        ),
    )
    parser.add_argument("--repop-start", type=int, default=0)
    parser.add_argument("--repop-end", type=int, default=499)
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=VALID_SCENARIOS,
        help="Comma-separated scenarios (default: fragile,resilient).",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--figure9-csv",
        type=Path,
        default=None,
        help=(
            "Optional Figure 9 combined CSV. Every overlapping catalogue is "
            "required to match its pre-cut counts exactly."
        ),
    )
    parser.add_argument(
        "--total-count-npz",
        type=Path,
        default=None,
        help=(
            "Optional 500-repopulation relative-J NPZ. Totals are compared "
            "and differences are reported, not treated as failures, because "
            "that diagnostic used a narrower validity mask."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repop_start < 0 or args.repop_end < args.repop_start:
        raise ValueError(
            "Require 0 <= --repop-start <= --repop-end."
        )
    if args.repop_end >= EXPECTED_REPOPULATIONS:
        raise ValueError("Requested repopulation IDs must lie within 0--499.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    theta_cut = round_up(
        healpix_pixel_size_deg(NSIDE), decimals=ROUND_UP_DECIMALS
    )
    if theta_cut != THETA_CUT_DEG:
        raise RuntimeError(
            "Production NSIDE-derived angular threshold changed: "
            f"expected {THETA_CUT_DEG}, found {theta_cut}."
        )


def catalogue_path(input_root: Path, repop_id: int, scenario: str) -> Path:
    return (
        input_root
        / f"repop_{repop_id:04d}"
        / f"fullrepop_hydro_{scenario}.h5"
    )


def decode_column_names(raw_names: Iterable[object]) -> list[str]:
    return [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in raw_names
    ]


def read_required_chunk(
    dataset: h5py.Dataset,
    start: int,
    stop: int,
    source_indices: dict[str, int],
) -> dict[str, np.ndarray]:
    """Read only columns needed by the production validity and dark masks."""
    ordered = sorted((index, name) for name, index in source_indices.items())
    source_columns = [index for index, _ in ordered]
    values = np.asarray(dataset[start:stop, source_columns], dtype=np.float64)
    return {
        name: values[:, local_index]
        for local_index, (_, name) in enumerate(ordered)
    }


def scan_catalogue(path: Path, chunk_size: int) -> dict[str, object]:
    """Scan one HDF5 catalogue once without maps, proxies, or brightness cuts."""
    n_physical_valid = 0
    n_vmax_excluded = 0
    n_pointlike = 0
    n_extended = 0

    with h5py.File(path, "r") as handle:
        dataset_name = f"iteration_{ITERATION}/data"
        if dataset_name not in handle:
            raise KeyError(f"Missing dataset {dataset_name!r} in {path}")
        dataset = handle[dataset_name]
        if dataset.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional dataset in {path}; "
                f"found shape {dataset.shape}."
            )
        if "column_names" not in dataset.attrs:
            raise KeyError(f"Missing column_names attribute in {path}")
        column_names = decode_column_names(dataset.attrs["column_names"])
        if len(column_names) != dataset.shape[1]:
            raise ValueError(
                f"column_names has {len(column_names)} entries for "
                f"{dataset.shape[1]} columns in {path}."
            )
        missing = [name for name in REQUIRED_COLUMNS if name not in column_names]
        if missing:
            raise KeyError(f"Missing required columns in {path}: {missing}")
        indices = {name: column_names.index(name) for name in REQUIRED_COLUMNS}
        n_input_rows = int(dataset.shape[0])

        for start in range(0, n_input_rows, chunk_size):
            stop = min(start + chunk_size, n_input_rows)
            values = read_required_chunk(dataset, start, stop, indices)
            valid = build_valid_mask(
                js=values["Js"],
                d_earth=values["D_Earth"],
                theta_s=values["theta_s"],
                r_s=values["r_s"],
                rho_s=values["rho_s"],
                x_e=values["Xearth"],
                y_e=values["Yearth"],
                z_e=values["Zearth"],
            )
            vmax = values["Vmax"]
            bad_vmax = valid & (~np.isfinite(vmax) | (vmax <= 0.0))
            if np.any(bad_vmax):
                row = start + int(np.flatnonzero(bad_vmax)[0])
                raise ValueError(
                    "Invalid Vmax for otherwise-valid halo at catalogue "
                    f"row {row} in {path}."
                )

            vmax_excluded = valid & (vmax > MAX_VMAX_KM_S)
            selected = valid & ~vmax_excluded
            pointlike = selected & (values["theta_s"] < THETA_CUT_DEG)
            extended = selected & (values["theta_s"] >= THETA_CUT_DEG)

            n_physical_valid += int(np.count_nonzero(valid))
            n_vmax_excluded += int(np.count_nonzero(vmax_excluded))
            n_pointlike += int(np.count_nonzero(pointlike))
            n_extended += int(np.count_nonzero(extended))

    n_total = n_pointlike + n_extended
    if n_total != n_physical_valid - n_vmax_excluded:
        raise RuntimeError(
            f"Count invariant failed for {path}: selected={n_total}, "
            f"physical_valid-vmax_excluded="
            f"{n_physical_valid - n_vmax_excluded}."
        )
    if n_total <= 0:
        raise RuntimeError(f"No selected dark subhalos found in {path}.")

    return {
        "n_input_rows": n_input_rows,
        "n_physical_valid": n_physical_valid,
        "n_vmax_excluded": n_vmax_excluded,
        "n_total": n_total,
        "n_pointlike": n_pointlike,
        "n_extended": n_extended,
    }


def build_output_row(
    scenario: str,
    repop_id: int,
    path: Path,
    chunk_size: int,
    counts: dict[str, object],
) -> dict[str, object]:
    n_total = int(counts["n_total"])
    n_pointlike = int(counts["n_pointlike"])
    n_extended = int(counts["n_extended"])
    if n_total != n_pointlike + n_extended:
        raise RuntimeError("n_total must equal n_pointlike + n_extended.")
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "repop_id": repop_id,
        "source_h5": str(path.resolve(strict=False)),
        "iteration": ITERATION,
        "n_input_rows": int(counts["n_input_rows"]),
        "n_physical_valid": int(counts["n_physical_valid"]),
        "n_vmax_excluded": int(counts["n_vmax_excluded"]),
        "n_total": n_total,
        "n_pointlike": n_pointlike,
        "n_extended": n_extended,
        "pointlike_fraction": n_pointlike / n_total,
        "extended_fraction": n_extended / n_total,
        "max_vmax_km_s": MAX_VMAX_KM_S,
        "vmax_selection": "inclusive",
        "nside": NSIDE,
        "theta_pix_deg": healpix_pixel_size_deg(NSIDE),
        "theta_cut_deg": THETA_CUT_DEG,
        "round_up_decimals": ROUND_UP_DECIMALS,
        "chunk_size": chunk_size,
    }


def _parse_int(row: dict[str, str], name: str, line: int) -> int:
    try:
        return int(row[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name} on line {line}: {row[name]!r}.") from error


def _parse_float(row: dict[str, str], name: str, line: int) -> float:
    try:
        value = float(row[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name} on line {line}: {row[name]!r}.") from error
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {name} on line {line}.")
    return value


def validate_checkpoint_row(row: dict[str, str], line: int) -> dict[str, object]:
    scenario = row["scenario"]
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Invalid scenario on line {line}: {scenario!r}.")
    parsed: dict[str, object] = dict(row)
    for name in (
        "schema_version",
        "repop_id",
        "iteration",
        "n_input_rows",
        "n_physical_valid",
        "n_vmax_excluded",
        "n_total",
        "n_pointlike",
        "n_extended",
        "nside",
        "round_up_decimals",
        "chunk_size",
    ):
        parsed[name] = _parse_int(row, name, line)
    for name in (
        "pointlike_fraction",
        "extended_fraction",
        "max_vmax_km_s",
        "theta_pix_deg",
        "theta_cut_deg",
    ):
        parsed[name] = _parse_float(row, name, line)

    if parsed["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version on line {line}.")
    repop_id = int(parsed["repop_id"])
    if not 0 <= repop_id < EXPECTED_REPOPULATIONS:
        raise ValueError(f"repop_id on line {line} must lie within 0--499.")
    if parsed["iteration"] != ITERATION:
        raise ValueError(f"Unexpected iteration on line {line}.")
    if parsed["vmax_selection"] != "inclusive":
        raise ValueError(f"Unexpected Vmax selection on line {line}.")
    if not np.isclose(parsed["max_vmax_km_s"], MAX_VMAX_KM_S):
        raise ValueError(f"Unexpected max_vmax_km_s on line {line}.")
    if parsed["nside"] != NSIDE or not np.isclose(
        parsed["theta_cut_deg"], THETA_CUT_DEG
    ):
        raise ValueError(f"Unexpected angular selection on line {line}.")
    if not np.isclose(
        parsed["theta_pix_deg"], healpix_pixel_size_deg(NSIDE)
    ) or parsed["round_up_decimals"] != ROUND_UP_DECIMALS:
        raise ValueError(f"Unexpected angular metadata on line {line}.")

    n_total = int(parsed["n_total"])
    n_pointlike = int(parsed["n_pointlike"])
    n_extended = int(parsed["n_extended"])
    n_input_rows = int(parsed["n_input_rows"])
    n_physical_valid = int(parsed["n_physical_valid"])
    n_vmax_excluded = int(parsed["n_vmax_excluded"])
    if min(
        n_input_rows,
        n_physical_valid,
        n_vmax_excluded,
        n_total,
        n_pointlike,
        n_extended,
    ) < 0:
        raise ValueError(f"Negative count on line {line}.")
    if n_physical_valid > n_input_rows or n_vmax_excluded > n_physical_valid:
        raise ValueError(f"Invalid input/selection counts on line {line}.")
    if n_total != n_pointlike + n_extended:
        raise ValueError(f"Population count invariant fails on line {line}.")
    if n_total <= 0:
        raise ValueError(f"n_total must be positive on line {line}.")
    if not np.isclose(parsed["pointlike_fraction"], n_pointlike / n_total):
        raise ValueError(f"pointlike_fraction is inconsistent on line {line}.")
    if not np.isclose(parsed["extended_fraction"], n_extended / n_total):
        raise ValueError(f"extended_fraction is inconsistent on line {line}.")
    if n_physical_valid - n_vmax_excluded != n_total:
        raise ValueError(f"Physical-selection count invariant fails on line {line}.")
    if int(parsed["chunk_size"]) <= 0:
        raise ValueError(f"chunk_size must be positive on line {line}.")
    if not row["source_h5"] or Path(row["source_h5"]).suffix.lower() not in {
        ".h5",
        ".hdf5",
    }:
        raise ValueError(f"Invalid source_h5 on line {line}.")
    return parsed


def load_checkpoint(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError(f"Checkpoint is not a regular file: {path}")
    records: dict[tuple[str, int], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ValueError(
                f"Checkpoint schema must be exactly {list(OUTPUT_COLUMNS)}."
            )
        for line, row in enumerate(reader, start=2):
            if None in row or set(row) != set(OUTPUT_COLUMNS):
                raise ValueError(f"Malformed checkpoint row on line {line}.")
            parsed = validate_checkpoint_row(row, line)
            key = (str(parsed["scenario"]), int(parsed["repop_id"]))
            if key in records:
                raise ValueError(f"Duplicate checkpoint key: {key}.")
            records[key] = parsed
    return records


def write_csv_atomic(
    path: Path,
    columns: tuple[str, ...],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
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
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def write_checkpoint(
    path: Path,
    records: dict[tuple[str, int], dict[str, object]],
) -> None:
    write_csv_atomic(path, OUTPUT_COLUMNS, (records[key] for key in sorted(records)))


def validate_resumed_source(record: dict[str, object], expected: Path) -> None:
    recorded = Path(str(record["source_h5"])).resolve(strict=False)
    expected = expected.resolve(strict=False)
    if recorded != expected:
        raise ValueError(
            f"Checkpoint source mismatch for {record['scenario']} "
            f"repop_{int(record['repop_id']):04d}: {recorded} != {expected}."
        )


def run_catalogues(
    args: argparse.Namespace,
    scan_catalogue_fn: Callable[[Path, int], dict[str, object]] = scan_catalogue,
) -> dict[tuple[str, int], dict[str, object]]:
    records = load_checkpoint(args.output_csv)
    figure9_counts = (
        None
        if getattr(args, "figure9_csv", None) is None
        else load_figure9_counts(args.figure9_csv)
    )
    for repop_id in range(args.repop_start, args.repop_end + 1):
        for scenario in args.scenarios:
            key = (scenario, repop_id)
            path = catalogue_path(args.input_root, repop_id, scenario)
            if key in records:
                validate_resumed_source(records[key], path)
                if figure9_counts is not None and key in figure9_counts:
                    validate_row_against_figure9_counts(
                        records[key], key, figure9_counts[key]
                    )
                print(f"Reusing {scenario} repop_{repop_id:04d}", flush=True)
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Catalogue not found: {path}")
            print(f"Scanning {scenario} repop_{repop_id:04d}: {path}", flush=True)
            counts = scan_catalogue_fn(path, args.chunk_size)
            new_row = build_output_row(
                scenario, repop_id, path, args.chunk_size, counts
            )
            if figure9_counts is not None and key in figure9_counts:
                validate_row_against_figure9_counts(
                    new_row, key, figure9_counts[key]
                )
            records[key] = new_row
            write_checkpoint(args.output_csv, records)
            print(
                f"  total={records[key]['n_total']:,}, "
                f"pointlike={records[key]['n_pointlike']:,}, "
                f"extended={records[key]['n_extended']:,}",
                flush=True,
            )
    return records


def requested_records(
    records: dict[tuple[str, int], dict[str, object]],
    scenarios: Iterable[str],
    repop_start: int,
    repop_end: int,
) -> list[dict[str, object]]:
    keys = [
        (scenario, repop_id)
        for scenario in scenarios
        for repop_id in range(repop_start, repop_end + 1)
    ]
    missing = [key for key in keys if key not in records]
    if missing:
        raise RuntimeError(f"Missing requested checkpoint rows: {missing[:10]}")
    return [records[key] for key in keys]


def summarize(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = list(rows)
    summaries: list[dict[str, object]] = []
    for scenario in VALID_SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        if not selected:
            continue
        selected.sort(key=lambda row: int(row["repop_id"]))
        totals = np.asarray([row["n_total"] for row in selected], dtype=np.int64)
        pointlike = np.asarray(
            [row["n_pointlike"] for row in selected], dtype=np.int64
        )
        extended = np.asarray(
            [row["n_extended"] for row in selected], dtype=np.int64
        )
        pointlike_fractions = pointlike / totals
        extended_fractions = extended / totals
        summaries.append(
            {
                "scenario": scenario,
                "n_repopulations": len(selected),
                "repop_start": int(selected[0]["repop_id"]),
                "repop_end": int(selected[-1]["repop_id"]),
                "mean_n_total": float(np.mean(totals)),
                "mean_n_pointlike": float(np.mean(pointlike)),
                "mean_n_extended": float(np.mean(extended)),
                "mean_pointlike_fraction": float(np.mean(pointlike_fractions)),
                "mean_extended_fraction": float(np.mean(extended_fractions)),
                "aggregate_pointlike_fraction": float(
                    pointlike.sum(dtype=np.int64) / totals.sum(dtype=np.int64)
                ),
                "aggregate_extended_fraction": float(
                    extended.sum(dtype=np.int64) / totals.sum(dtype=np.int64)
                ),
                "max_vmax_km_s": MAX_VMAX_KM_S,
                "vmax_selection": "inclusive",
                "nside": NSIDE,
                "theta_cut_deg": THETA_CUT_DEG,
            }
        )
    return summaries


def load_figure9_counts(path: Path) -> dict[tuple[str, int], tuple[int, int, int]]:
    counts: dict[tuple[str, int], tuple[int, int, int]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = FIGURE9_REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Figure 9 CSV is missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            scenario = row["scenario"]
            if scenario not in VALID_SCENARIOS:
                raise ValueError(f"Invalid Figure 9 scenario on line {line}.")
            repop_id = _parse_int(row, "repop_id", line)
            if not np.isclose(_parse_float(row, "max_vmax_km_s", line), MAX_VMAX_KM_S):
                continue
            if not np.isclose(_parse_float(row, "theta_min_deg", line), THETA_CUT_DEG):
                continue
            value = (
                _parse_int(row, "n_valid_total", line),
                _parse_int(row, "n_pointlike_total", line),
                _parse_int(row, "n_extended_total", line),
            )
            key = (scenario, repop_id)
            if key in counts and counts[key] != value:
                raise ValueError(f"Inconsistent duplicate Figure 9 counts for {key}.")
            counts[key] = value
    return counts


def validate_against_figure9(
    rows: Iterable[dict[str, object]], path: Path
) -> int:
    expected = load_figure9_counts(path)
    compared = 0
    for row in rows:
        key = (str(row["scenario"]), int(row["repop_id"]))
        if key not in expected:
            continue
        validate_row_against_figure9_counts(row, key, expected[key])
        compared += 1
    if compared == 0:
        raise ValueError("No requested catalogue overlaps the Figure 9 CSV.")
    return compared


def validate_row_against_figure9_counts(
    row: dict[str, object],
    key: tuple[str, int],
    expected: tuple[int, int, int],
) -> None:
    actual = (
        int(row["n_total"]),
        int(row["n_pointlike"]),
        int(row["n_extended"]),
    )
    if actual != expected:
        raise ValueError(
            f"Figure 9 count mismatch for {key}: {actual} != {expected}."
        )


def compare_total_count_npz(
    rows: Iterable[dict[str, object]], path: Path
) -> tuple[int, list[tuple[str, int, int, int]]]:
    compared = 0
    mismatches: list[tuple[str, int, int, int]] = []
    with np.load(path, allow_pickle=False) as data:
        if not np.isclose(float(data["max_vmax_km_s"]), MAX_VMAX_KM_S):
            raise ValueError("Total-count NPZ does not use Vmax <= 8 km/s.")
        lookups: dict[str, dict[int, int]] = {}
        for scenario in VALID_SCENARIOS:
            ids = np.asarray(data[f"{scenario}_repop_ids"], dtype=np.int64)
            totals = np.asarray(data[f"{scenario}_n_total"], dtype=np.int64)
            if ids.shape != totals.shape or len(np.unique(ids)) != len(ids):
                raise ValueError(f"Invalid {scenario} arrays in total-count NPZ.")
            lookups[scenario] = dict(zip(ids.tolist(), totals.tolist()))
        for row in rows:
            scenario = str(row["scenario"])
            repop_id = int(row["repop_id"])
            if repop_id not in lookups[scenario]:
                continue
            strict_total = int(row["n_total"])
            diagnostic_total = lookups[scenario][repop_id]
            compared += 1
            if strict_total != diagnostic_total:
                mismatches.append(
                    (scenario, repop_id, strict_total, diagnostic_total)
                )
    if compared == 0:
        raise ValueError("No requested catalogue overlaps the total-count NPZ.")
    return compared, mismatches


def print_summary(summaries: Iterable[dict[str, object]]) -> None:
    print("\nDark population summary (before brightness cuts)")
    for row in summaries:
        print(
            f"  {row['scenario']}: N={row['n_repopulations']}, "
            f"mean total={float(row['mean_n_total']):,.3f}, "
            f"mean pointlike={float(row['mean_n_pointlike']):,.3f} "
            f"({float(row['mean_pointlike_fraction']):.8%}), "
            f"mean extended={float(row['mean_n_extended']):,.3f} "
            f"({float(row['mean_extended_fraction']):.8%})"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.summary_csv is None:
        args.summary_csv = args.output_csv.with_name(
            f"{args.output_csv.stem}_summary.csv"
        )

    records = run_catalogues(args)
    selected = requested_records(
        records, args.scenarios, args.repop_start, args.repop_end
    )

    if args.figure9_csv is not None:
        compared = validate_against_figure9(selected, args.figure9_csv)
        print(f"Figure 9 validation passed for {compared} catalogue(s).")

    if args.total_count_npz is not None:
        compared, mismatches = compare_total_count_npz(
            selected, args.total_count_npz
        )
        if mismatches:
            print(
                f"Total-count NPZ comparison: {len(mismatches)}/{compared} "
                "differ. This can occur because the NPZ checks only Js and "
                "Vmax, while this diagnostic uses the full production "
                "validity mask."
            )
            for mismatch in mismatches[:10]:
                print(
                    "  scenario={}, repop_id={}, production_valid={}, "
                    "relative_J_valid={}".format(*mismatch)
                )
        else:
            print(f"Total-count NPZ validation passed for {compared} catalogue(s).")

    summaries = summarize(selected)
    write_csv_atomic(args.summary_csv, SUMMARY_COLUMNS, summaries)
    print_summary(summaries)
    print(f"Per-repopulation CSV: {args.output_csv}")
    print(f"Scenario summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
