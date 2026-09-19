#!/usr/bin/env python3
"""Plot Vmax for the subhalo defining D_min in each repopulation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile
from typing import Iterable

import h5py
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_REPOPULATIONS = 500
VALID_SCENARIOS = ("fragile", "resilient")
SCENARIO_COLORS = {
    "fragile": "#EE3377",
    "resilient": "#009988",
}
REQUIRED_CSV_COLUMNS = {"scenario", "repop_id", "min_dgc_kpc"}
REQUIRED_DATA_COLUMNS = {"Vmax", "Xearth", "Yearth", "Zearth"}
OUTPUT_CSV_COLUMNS = (
    "scenario",
    "repop_id",
    "source_h5",
    "dmin_kpc",
    "vmax_kms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Vmax for the unique subhalo at minimum galactocentric "
            "distance in each 500-repopulation catalogue."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs",
        help=(
            "Directory containing repop_XXXX/fullrepop_hydro_<scenario>.h5 "
            f"files (default: {REPOSITORY_ROOT / 'outputs'})."
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs"
        / "diagnostics"
        / "full_catalogue_dgc_0500.csv",
        help="CSV containing scenario, repop_id, and min_dgc_kpc.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs"
        / "diagnostics"
        / "dmin_vmax_distribution.csv",
        help=(
            "Checkpoint CSV for scenario, repopulation, source HDF5, D_min, "
            "and Vmax values."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "diagnostics"
        / "catalogue"
        / "plots"
        / "dmin_vmax_distribution",
        help="Directory for dmin_vmax_distribution.png and .pdf.",
    )
    parser.add_argument(
        "--scenario",
        choices=VALID_SCENARIOS,
        default="resilient",
        help="Disruption scenario to process (default: resilient).",
    )
    parser.add_argument(
        "--repop-start",
        type=int,
        default=0,
        help="First repopulation ID to process (default: 0).",
    )
    parser.add_argument(
        "--n-repops",
        type=int,
        default=EXPECTED_REPOPULATIONS,
        help="Number of consecutive repopulations to process (default: 500).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=30,
        help="Number of linear Vmax histogram bins (default: 30).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of HDF5 rows read per chunk (default: 1000000).",
    )
    parser.add_argument(
        "--dmin-rtol",
        type=float,
        default=1.0e-10,
        help="Relative tolerance for the HDF5/CSV D_min check.",
    )
    parser.add_argument(
        "--dmin-atol-kpc",
        type=float,
        default=1.0e-10,
        help="Absolute tolerance in kpc for the HDF5/CSV D_min check.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--skip-plot",
        action="store_true",
        help="Scan and checkpoint CSV values without initializing or saving a plot.",
    )
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="Read the checkpoint CSV only and write the Vmax histogram.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repop_start < 0:
        raise ValueError("--repop-start must be non-negative.")
    if args.n_repops <= 0:
        raise ValueError("--n-repops must be positive.")
    if args.repop_start + args.n_repops > EXPECTED_REPOPULATIONS:
        raise ValueError(
            "Requested repopulation interval must lie within IDs 0--499."
        )
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if not np.isfinite(args.dmin_rtol) or args.dmin_rtol < 0.0:
        raise ValueError("--dmin-rtol must be finite and non-negative.")
    if not np.isfinite(args.dmin_atol_kpc) or args.dmin_atol_kpc < 0.0:
        raise ValueError("--dmin-atol-kpc must be finite and non-negative.")


def catalogue_path(input_root: Path, repop_id: int, scenario: str) -> Path:
    return (
        input_root
        / f"repop_{repop_id:04d}"
        / f"fullrepop_hydro_{scenario}.h5"
    )


def make_repop_ids(repop_start: int, n_repops: int) -> range:
    """Return the validated consecutive repopulation interval."""
    return range(repop_start, repop_start + n_repops)


def validate_catalogue_files(
    input_root: Path,
    scenario: str,
    repop_ids: Iterable[int],
) -> None:
    """Require one HDF5 catalogue for every requested repopulation."""
    paths = [
        catalogue_path(input_root, repop_id, scenario) for repop_id in repop_ids
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        remaining = len(missing) - min(len(missing), 10)
        suffix = "" if remaining == 0 else f"\n  ... and {remaining} more"
        raise FileNotFoundError(
            f"Scenario {scenario!r} is missing {len(missing)} requested "
            f"HDF5 catalogue(s):\n{preview}{suffix}"
        )


def load_dmin_values(
    path: Path,
    scenario: str,
    repop_ids: Iterable[int],
) -> dict[int, float]:
    """Read one D_min for each requested repopulation and scenario."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    requested_ids = set(repop_ids)
    dmin_by_repop: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_CSV_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Input CSV is missing columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            if row["scenario"] != scenario:
                continue
            try:
                repop_id = int(row["repop_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {scenario} row on line {line_number}: {row}"
                ) from error
            if repop_id not in requested_ids:
                continue
            try:
                dmin = float(row["min_dgc_kpc"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {scenario} row on line {line_number}: {row}"
                ) from error
            if repop_id in dmin_by_repop:
                raise ValueError(
                    f"Duplicate {scenario} repop_id in CSV: {repop_id}"
                )
            if not np.isfinite(dmin) or dmin <= 0.0:
                raise ValueError(
                    f"min_dgc_kpc must be finite and positive on line "
                    f"{line_number}."
                )
            dmin_by_repop[repop_id] = dmin

    expected_ids = requested_ids
    actual_ids = set(dmin_by_repop)
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)
        details: list[str] = []
        if missing_ids:
            details.append(f"missing IDs: {missing_ids[:10]}")
        raise ValueError(
            f"Scenario {scenario!r} must contain exactly "
            f"{len(expected_ids)} requested repopulation(s); "
            + "; ".join(details)
        )
    return dmin_by_repop


def parse_output_repop_id(value: str, line_number: int) -> int:
    """Parse and validate one repopulation ID in the checkpoint CSV."""
    try:
        repop_id = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid repop_id on line {line_number}: {value!r}."
        ) from error
    if not 0 <= repop_id < EXPECTED_REPOPULATIONS:
        raise ValueError(
            f"repop_id on line {line_number} must lie within IDs 0--499."
        )
    return repop_id


def parse_positive_output_float(
    value: str,
    column: str,
    line_number: int,
) -> float:
    """Parse a finite, strictly positive checkpoint value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {column} on line {line_number}: {value!r}."
        ) from error
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(
            f"{column} on line {line_number} must be finite and positive."
        )
    return parsed


def validate_source_h5(source_h5: str, line_number: int) -> str:
    """Require a non-empty HDF5 source path without requiring it to exist."""
    if not source_h5 or source_h5 != source_h5.strip():
        raise ValueError(
            f"source_h5 on line {line_number} must be a non-empty path."
        )
    if Path(source_h5).suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(
            f"source_h5 on line {line_number} must name an HDF5 file."
        )
    return source_h5


def load_output_records(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    """Load a fully validated Vmax checkpoint CSV, if it already exists."""
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError(f"Checkpoint CSV is not a regular file: {path}")

    records: dict[tuple[str, int], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Checkpoint CSV has no header: {path}")
        if tuple(fieldnames) != OUTPUT_CSV_COLUMNS:
            raise ValueError(
                "Checkpoint CSV schema must be exactly "
                f"{list(OUTPUT_CSV_COLUMNS)}; found {fieldnames}."
            )

        for line_number, row in enumerate(reader, start=2):
            if None in row or set(row) != set(OUTPUT_CSV_COLUMNS):
                raise ValueError(
                    f"Checkpoint CSV has malformed fields on line {line_number}."
                )
            scenario = row["scenario"]
            if scenario not in VALID_SCENARIOS:
                raise ValueError(
                    f"Invalid scenario on line {line_number}: {scenario!r}."
                )
            repop_id = parse_output_repop_id(row["repop_id"], line_number)
            source_h5 = validate_source_h5(row["source_h5"], line_number)
            dmin_kpc = parse_positive_output_float(
                row["dmin_kpc"], "dmin_kpc", line_number
            )
            vmax_kms = parse_positive_output_float(
                row["vmax_kms"], "vmax_kms", line_number
            )
            key = (scenario, repop_id)
            if key in records:
                raise ValueError(
                    "Duplicate checkpoint key for "
                    f"scenario={scenario!r}, repop_id={repop_id}."
                )
            records[key] = {
                "scenario": scenario,
                "repop_id": repop_id,
                "source_h5": source_h5,
                "dmin_kpc": dmin_kpc,
                "vmax_kms": vmax_kms,
            }
    return records


def write_output_records_atomic(
    path: Path,
    records: dict[tuple[str, int], dict[str, object]],
) -> None:
    """Atomically replace the checkpoint CSV after a validated repopulation."""
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
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_CSV_COLUMNS)
            writer.writeheader()
            for key in sorted(records):
                writer.writerow(records[key])
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


def validate_completed_source(
    record: dict[str, object],
    expected_path: Path,
) -> None:
    """Ensure a resumed checkpoint still refers to this scan's HDF5 file."""
    recorded_path = Path(str(record["source_h5"])).resolve(strict=False)
    expected_path = expected_path.resolve(strict=False)
    if recorded_path != expected_path:
        raise ValueError(
            "Checkpoint source_h5 does not match the requested catalogue for "
            f"scenario={record['scenario']!r}, repop_id={record['repop_id']}: "
            f"{recorded_path} != {expected_path}."
        )


def decode_column_names(raw_names: np.ndarray) -> list[str]:
    return [
        name.decode() if isinstance(name, bytes) else str(name)
        for name in raw_names
    ]


def read_earth_position(handle: h5py.File, path: Path) -> np.ndarray:
    """Read the observer position saved alongside this catalogue."""
    base_location = "inputs/host/position_Earth"
    unit_location = f"{base_location}/unit"
    value_location = f"{base_location}/value"
    if unit_location not in handle:
        raise KeyError(f"Expected dataset {unit_location!r} not found in {path}")
    if value_location not in handle:
        raise KeyError(f"Expected group {value_location!r} not found in {path}")

    unit_dataset = handle[unit_location]
    if not isinstance(unit_dataset, h5py.Dataset):
        raise ValueError(f"Expected dataset {unit_location!r} in {path}")
    raw_unit = unit_dataset[()]
    unit = raw_unit.decode() if isinstance(raw_unit, bytes) else str(raw_unit)
    if unit != "kpc":
        raise ValueError(
            f"Unexpected observer-position unit in {path}: "
            f"expected 'kpc', found {unit!r}."
        )

    value_group = handle[value_location]
    if not isinstance(value_group, h5py.Group):
        raise ValueError(f"Expected group {value_location!r} in {path}")

    values = []
    for component in range(3):
        item_location = f"{value_location}/item_{component}"
        if item_location not in handle:
            raise KeyError(
                f"Expected dataset {item_location!r} not found in {path}"
            )
        item_dataset = handle[item_location]
        if not isinstance(item_dataset, h5py.Dataset):
            raise ValueError(f"Expected dataset {item_location!r} in {path}")
        raw_value = np.asarray(item_dataset[...], dtype=np.float64)
        if raw_value.shape != () or not np.isfinite(raw_value.item()):
            raise ValueError(
                f"Invalid observer-position component {component} in {path}: "
                "expected one finite scalar in kpc."
            )
        values.append(raw_value.item())

    position = np.asarray(values, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(
            f"Invalid observer position in {path}: expected three finite kpc values."
        )
    return position


def galactocentric_distance_kpc(
    xearth: np.ndarray,
    yearth: np.ndarray,
    zearth: np.ndarray,
    earth_position_kpc: np.ndarray,
) -> np.ndarray:
    """Recover D_GC from the Earth-centred convention written by the pipeline.

    The HDF5 writer stores Xearth = -(Xgc - Xearth_position),
    Yearth = Ygc - Yearth_position, and Zearth = Zgc - Zearth_position.
    """
    xgc = earth_position_kpc[0] - xearth
    ygc = earth_position_kpc[1] + yearth
    zgc = earth_position_kpc[2] + zearth
    return np.sqrt(xgc**2 + ygc**2 + zgc**2)


def find_dmin_subhalo_vmax(path: Path, chunk_size: int) -> tuple[float, float]:
    """Return (D_min, Vmax) after scanning one HDF5 catalogue in chunks."""
    best_dgc = np.inf
    best_vmax = np.nan
    n_at_best = 0

    with h5py.File(path, "r") as handle:
        dataset_name = "iteration_0/data"
        if dataset_name not in handle:
            raise KeyError(f"Expected dataset {dataset_name!r} not found in {path}")
        dataset = handle[dataset_name]
        if dataset.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional dataset in {path}; found "
                f"shape {dataset.shape}."
            )
        if "column_names" not in dataset.attrs:
            raise KeyError(f"Missing column_names attribute in {path}")
        column_names = decode_column_names(dataset.attrs["column_names"])
        missing = sorted(REQUIRED_DATA_COLUMNS - set(column_names))
        if missing:
            raise ValueError(f"Dataset in {path} is missing columns: {missing}")
        if dataset.shape[1] != len(column_names):
            raise ValueError(
                f"Column metadata does not match dataset shape in {path}: "
                f"{len(column_names)} names for {dataset.shape[1]} columns."
            )

        indices = {name: column_names.index(name) for name in REQUIRED_DATA_COLUMNS}
        earth_position = read_earth_position(handle, path)
        n_rows = int(dataset.shape[0])
        if n_rows == 0:
            raise ValueError(f"No subhalos stored in {path}")

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            chunk = np.asarray(dataset[start:stop, :], dtype=np.float64)
            dgc = galactocentric_distance_kpc(
                chunk[:, indices["Xearth"]],
                chunk[:, indices["Yearth"]],
                chunk[:, indices["Zearth"]],
                earth_position,
            )
            if np.any(~np.isfinite(dgc)):
                raise ValueError(f"Non-finite reconstructed D_GC values in {path}")

            local_best = float(np.min(dgc))
            local_winners = np.flatnonzero(dgc == local_best)
            if local_best < best_dgc:
                best_dgc = local_best
                best_vmax = float(chunk[local_winners[0], indices["Vmax"]])
                n_at_best = int(local_winners.size)
            elif local_best == best_dgc:
                n_at_best += int(local_winners.size)

    if n_at_best != 1:
        raise RuntimeError(
            f"Expected one D_min subhalo in {path}; found {n_at_best}."
        )
    if not np.isfinite(best_vmax) or best_vmax <= 0.0:
        raise ValueError(
            f"D_min subhalo in {path} has invalid Vmax={best_vmax!r}."
        )
    return best_dgc, best_vmax


def make_linear_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return padded linear bin edges, including the degenerate-data case."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = 0.05 * (upper - lower)
    if padding == 0.0:
        padding = max(abs(lower), 1.0) * 1.0e-6
    return np.linspace(lower - padding, upper + padding, n_bins + 1)


def plot_vmax_distribution(
    values: np.ndarray,
    scenario: str,
    output_dir: Path,
    n_bins: int,
    dpi: int,
) -> list[Path]:
    """Draw and save the absolute-count Vmax histogram."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15.5,
        }
    )
    figure, axis = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    axis.hist(
        values,
        bins=make_linear_bins(values, n_bins),
        density=False,
        histtype="stepfilled",
        color=SCENARIO_COLORS[scenario],
        alpha=0.78,
    )
    axis.set_title(f"{scenario.capitalize()} scenario")
    axis.set_xlabel(r"$V_{\max}\,[\mathrm{km\ s^{-1}}]$")
    axis.set_ylabel("Number of repopulations")
    axis.set_ylim(bottom=0)
    axis.tick_params(axis="both", which="both", labelsize=15)
    axis.grid(False, which="both")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for output_format in ("png", "pdf"):
        output_path = output_dir / f"dmin_vmax_distribution.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = dpi
        figure.savefig(output_path, **save_kwargs)
        saved.append(output_path)
        print(f"Saved: {output_path}")
    plt.close(figure)
    return saved


def requested_records(
    records: dict[tuple[str, int], dict[str, object]],
    scenario: str,
    repop_ids: Iterable[int],
) -> list[dict[str, object]]:
    """Return every requested checkpoint record, rejecting incomplete scans."""
    selected: list[dict[str, object]] = []
    missing_ids: list[int] = []
    for repop_id in repop_ids:
        record = records.get((scenario, repop_id))
        if record is None:
            missing_ids.append(repop_id)
        else:
            selected.append(record)
    if missing_ids:
        raise ValueError(
            f"Checkpoint CSV is missing {len(missing_ids)} requested "
            f"{scenario!r} repopulation(s): {missing_ids[:10]}."
        )
    return selected


def scan_catalogues(
    args: argparse.Namespace,
    repop_ids: range,
    records: dict[tuple[str, int], dict[str, object]],
) -> None:
    """Scan only missing catalogues, validating and checkpointing each one."""
    pending_ids: list[int] = []
    for repop_id in repop_ids:
        existing = records.get((args.scenario, repop_id))
        expected_path = catalogue_path(args.input_root, repop_id, args.scenario)
        if existing is None:
            pending_ids.append(repop_id)
        else:
            validate_completed_source(existing, expected_path)

    if not pending_ids:
        print("All requested repopulations are already checkpointed.")
        return

    csv_dmins = load_dmin_values(args.input_csv, args.scenario, pending_ids)
    validate_catalogue_files(args.input_root, args.scenario, pending_ids)
    for sequence, repop_id in enumerate(pending_ids, start=1):
        path = catalogue_path(args.input_root, repop_id, args.scenario)
        hdf5_dmin, vmax = find_dmin_subhalo_vmax(path, args.chunk_size)
        csv_dmin = csv_dmins[repop_id]
        if not np.isclose(
            hdf5_dmin,
            csv_dmin,
            rtol=args.dmin_rtol,
            atol=args.dmin_atol_kpc,
        ):
            raise RuntimeError(
                f"D_min mismatch for repop_{repop_id:04d}: HDF5="
                f"{hdf5_dmin:.17e} kpc, CSV={csv_dmin:.17e} kpc "
                f"(rtol={args.dmin_rtol:g}, atol={args.dmin_atol_kpc:g} kpc)."
            )
        records[(args.scenario, repop_id)] = {
            "scenario": args.scenario,
            "repop_id": repop_id,
            "source_h5": str(path.resolve(strict=False)),
            "dmin_kpc": hdf5_dmin,
            "vmax_kms": vmax,
        }
        write_output_records_atomic(args.output_csv, records)
        print(
            f"[{sequence:03d}/{len(pending_ids):03d}] "
            f"repop_{repop_id:04d} {args.scenario}: "
            f"D_min={hdf5_dmin:.8g} kpc | Vmax={vmax:.8g} km s^-1",
            flush=True,
        )


def run(args: argparse.Namespace) -> None:
    """Run either the resumable HDF5 scan or the CSV-only plotting stage."""
    repop_ids = make_repop_ids(args.repop_start, args.n_repops)
    records = load_output_records(args.output_csv)
    if not args.plot_only:
        scan_catalogues(args, repop_ids, records)
    if args.skip_plot:
        return

    selected = requested_records(records, args.scenario, repop_ids)
    values = np.asarray(
        [record["vmax_kms"] for record in selected],
        dtype=np.float64,
    )
    if values.shape != (len(repop_ids),):
        raise RuntimeError(
            f"Expected {len(repop_ids)} Vmax values; got {values.size}."
        )
    plot_vmax_distribution(
        values=values,
        scenario=args.scenario,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        dpi=args.dpi,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
