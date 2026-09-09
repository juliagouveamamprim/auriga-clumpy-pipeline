#!/usr/bin/env python3

"""Save the maximum valid Js value from each repopulated catalogue."""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
VALID_SCENARIOS = ("fragile", "resilient")
REPOP_IDS = range(500)
CSV_FIELDS = ("scenario", "repop_id", "n_saved", "js_max")
DEFAULT_OUTPUT_CSV = (
    REPOSITORY_ROOT
    / "outputs"
    / "diagnostics"
    / "repop_jsmax_diagnostic_all_repopulations.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find the maximum positive, finite Js in repopulations "
            "0000--0499 for the fragile and resilient scenarios."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Directory containing repop_XXXX/fullrepop_hydro_"
            "<scenario>.h5 files."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of HDF5 rows read per chunk.",
    )
    return parser.parse_args()


def catalogue_path(input_root, repop_id, scenario):
    return (
        input_root
        / f"repop_{repop_id:04d}"
        / f"fullrepop_hydro_{scenario}.h5"
    )


def validate_inputs(input_root):
    missing = []

    for scenario in VALID_SCENARIOS:
        for repop_id in REPOP_IDS:
            path = catalogue_path(input_root, repop_id, scenario)
            if not path.is_file():
                missing.append(path)

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        remaining = len(missing) - min(len(missing), 10)
        suffix = "" if remaining == 0 else f"\n  ... and {remaining} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} input catalogue(s):\n"
            f"{preview}{suffix}"
        )


def scan_catalogue(path, chunk_size):
    js_max = -np.inf

    with h5py.File(path, "r") as handle:
        if "iteration_0/data" not in handle:
            raise KeyError(
                f"Expected dataset 'iteration_0/data' not found in {path}"
            )

        dataset = handle["iteration_0/data"]
        if dataset.ndim != 2 or dataset.shape[1] < 1:
            raise ValueError(
                f"Expected a two-dimensional dataset with a Js column "
                f"in {path}; found shape {dataset.shape}."
            )

        n_rows = dataset.shape[0]
        n_saved = int(n_rows)

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            js = dataset[start:stop, 0]
            good = np.isfinite(js) & (js > 0.0)
            js = js[good]

            if js.size == 0:
                continue

            js_max = max(js_max, float(np.max(js)))

    if not np.isfinite(js_max):
        raise RuntimeError(f"No positive, finite Js values found in {path}")

    return n_saved, js_max


def load_checkpoint(path):
    completed = {}

    if not path.exists() or path.stat().st_size == 0:
        return completed

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                f"Unexpected columns in checkpoint {path}: "
                f"{reader.fieldnames}; expected {list(CSV_FIELDS)}."
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                scenario = row["scenario"]
                repop_id = int(row["repop_id"])
                n_saved = int(row["n_saved"])
                js_max = float(row["js_max"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid checkpoint row at {path}:{line_number}: {row}"
                ) from error

            key = (scenario, repop_id)
            if scenario not in VALID_SCENARIOS or repop_id not in REPOP_IDS:
                raise ValueError(
                    f"Unexpected catalogue at {path}:{line_number}: {key}"
                )
            if key in completed:
                raise ValueError(
                    f"Duplicate catalogue at {path}:{line_number}: {key}"
                )
            if n_saved <= 0 or not np.isfinite(js_max) or js_max <= 0.0:
                raise ValueError(
                    f"Invalid diagnostic values at {path}:{line_number}: "
                    f"n_saved={n_saved}, js_max={js_max}"
                )

            completed[key] = {
                "n_saved": n_saved,
                "js_max": js_max,
            }

    return completed


def append_checkpoint(path, scenario, repop_id, n_saved, js_max):
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "scenario": scenario,
                "repop_id": repop_id,
                "n_saved": n_saved,
                "js_max": f"{js_max:.17e}",
            }
        )


def main():
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    validate_inputs(args.input_root)
    completed = load_checkpoint(DEFAULT_OUTPUT_CSV)
    DEFAULT_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    n_catalogues = len(VALID_SCENARIOS) * len(REPOP_IDS)
    sequence = 0

    for scenario in VALID_SCENARIOS:
        for repop_id in REPOP_IDS:
            sequence += 1
            key = (scenario, repop_id)

            if key in completed:
                result = completed[key]
                status = "checkpoint"
            else:
                path = catalogue_path(args.input_root, repop_id, scenario)
                n_saved, js_max = scan_catalogue(path, args.chunk_size)
                append_checkpoint(
                    DEFAULT_OUTPUT_CSV,
                    scenario,
                    repop_id,
                    n_saved,
                    js_max,
                )
                result = {"n_saved": n_saved, "js_max": js_max}
                completed[key] = result
                status = "saved"

            print(
                f"[{sequence:04d}/{n_catalogues:04d}] "
                f"repop_{repop_id:04d} {scenario}: {status} | "
                f"n_saved={result['n_saved']:,} | "
                f"js_max={result['js_max']:.6e}",
                flush=True,
            )

    print("\nUnweighted mean of the 500 catalogue maxima:")
    for scenario in VALID_SCENARIOS:
        values = np.asarray(
            [
                completed[(scenario, repop_id)]["js_max"]
                for repop_id in REPOP_IDS
            ],
            dtype=np.float64,
        )
        print(
            f"  {scenario}: {np.mean(values):.6e} GeV^2 cm^-5",
            flush=True,
        )

    print(f"\nSaved: {DEFAULT_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
