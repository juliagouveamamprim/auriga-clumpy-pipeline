#!/usr/bin/env python3

"""Build per-catalogue cumulative distributions of relative subhalo Js."""

import argparse
from pathlib import Path

import h5py
import numpy as np


VALID_SCENARIOS = ("fragile", "resilient")
LOG_REL_MIN = -16.0
LOG_REL_MAX = 0.0
N_BINS_REL = 320


def parse_scenarios(text):
    scenarios = tuple(
        scenario.strip().lower()
        for scenario in text.split(",")
        if scenario.strip()
    )

    if not scenarios:
        raise argparse.ArgumentTypeError("At least one scenario is required.")

    invalid = sorted(set(scenarios) - set(VALID_SCENARIOS))
    if invalid:
        raise argparse.ArgumentTypeError(
            "Unknown scenario(s): " + ", ".join(invalid)
        )

    if len(set(scenarios)) != len(scenarios):
        raise argparse.ArgumentTypeError("Scenarios must not be repeated.")

    return scenarios


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute one cumulative distribution of Js/Js,max per "
            "repopulated catalogue."
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
        "--output-npz",
        type=Path,
        required=True,
        help="Compressed output containing the per-catalogue CDFs.",
    )
    parser.add_argument(
        "--repop-start",
        type=int,
        default=0,
        help="First repopulation index to process.",
    )
    parser.add_argument(
        "--n-repops",
        type=int,
        default=500,
        help="Number of consecutive repopulations per scenario.",
    )
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=VALID_SCENARIOS,
        help="Comma-separated scenarios (default: fragile,resilient).",
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


def validate_args(args):
    if args.repop_start < 0:
        raise ValueError("--repop-start must be non-negative.")
    if args.n_repops <= 0:
        raise ValueError("--n-repops must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.output_npz.suffix != ".npz":
        raise ValueError("--output-npz must end in .npz.")


def validate_inputs(input_root, repop_ids, scenarios):
    missing = []

    for scenario in scenarios:
        for repop_id in repop_ids:
            path = catalogue_path(input_root, repop_id, scenario)
            if not path.is_file():
                missing.append(path)

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        remaining = len(missing) - min(len(missing), 10)
        suffix = "" if remaining == 0 else f"\n  ... and {remaining} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} input catalogue(s):\n{preview}{suffix}"
        )


def find_max_logj(path, chunk_size):
    n_total = 0
    n_bad = 0
    max_logj = -np.inf

    with h5py.File(path, "r") as handle:
        dataset = handle["iteration_0/data"]
        n_rows = dataset.shape[0]

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            js = dataset[start:stop, 0]

            good = np.isfinite(js) & (js > 0.0)
            n_bad += int(np.count_nonzero(~good))
            js = js[good]

            if js.size == 0:
                continue

            logj = np.log10(js)
            max_logj = max(max_logj, float(np.max(logj)))
            n_total += int(logj.size)

    if not np.isfinite(max_logj):
        raise RuntimeError(f"No valid Js values found in {path}")

    return max_logj, n_total, n_bad


def process_catalogue(path, chunk_size, relative_edges):
    max_logj, n_total, n_bad = find_max_logj(path, chunk_size)
    histogram = np.zeros(len(relative_edges) - 1, dtype=np.int64)
    n_under = 0
    n_over = 0

    with h5py.File(path, "r") as handle:
        dataset = handle["iteration_0/data"]
        n_rows = dataset.shape[0]

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            js = dataset[start:stop, 0]
            good = np.isfinite(js) & (js > 0.0)
            js = js[good]

            if js.size == 0:
                continue

            log_relative_js = np.log10(js) - max_logj
            n_under += int(
                np.count_nonzero(log_relative_js < relative_edges[0])
            )
            n_over += int(
                np.count_nonzero(log_relative_js > relative_edges[-1])
            )

            in_range = (
                (log_relative_js >= relative_edges[0])
                & (log_relative_js <= relative_edges[-1])
            )
            histogram += np.histogram(
                log_relative_js[in_range],
                bins=relative_edges,
            )[0]

    counted = n_under + int(histogram.sum()) + n_over
    if counted != n_total:
        raise RuntimeError(
            f"Histogram count mismatch for {path}: "
            f"counted {counted}, expected {n_total}."
        )

    cumulative_counts = np.concatenate(
        ([n_under], n_under + np.cumsum(histogram, dtype=np.int64))
    )
    cdf = cumulative_counts.astype(np.float64) / n_total

    return {
        "cdf": cdf,
        "n_total": n_total,
        "max_log10_js": max_logj,
        "n_bad": n_bad,
        "n_under": n_under,
        "n_over": n_over,
    }


def main():
    args = parse_args()
    validate_args(args)

    repop_ids = np.arange(
        args.repop_start,
        args.repop_start + args.n_repops,
        dtype=np.int64,
    )
    validate_inputs(args.input_root, repop_ids, args.scenarios)

    relative_edges = np.linspace(
        LOG_REL_MIN,
        LOG_REL_MAX,
        N_BINS_REL + 1,
    )
    x_values = 10.0**relative_edges

    output = {
        "xvals": x_values,
        "log_relative_edges": relative_edges,
        "scenarios": np.asarray(args.scenarios),
        "repop_start": np.asarray(args.repop_start),
        "n_repops": np.asarray(args.n_repops),
        "chunk_size": np.asarray(args.chunk_size),
    }

    for scenario in args.scenarios:
        print(f"\n=== {scenario.upper()} ===", flush=True)
        cdfs = []
        n_total = []
        max_log10_js = []
        n_bad = []
        n_under = []
        n_over = []

        for sequence, repop_id in enumerate(repop_ids, start=1):
            path = catalogue_path(args.input_root, repop_id, scenario)
            print(
                f"[{sequence:03d}/{args.n_repops:03d}] "
                f"repop_{repop_id:04d} {scenario}",
                flush=True,
            )
            result = process_catalogue(
                path,
                args.chunk_size,
                relative_edges,
            )
            cdfs.append(result["cdf"])
            n_total.append(result["n_total"])
            max_log10_js.append(result["max_log10_js"])
            n_bad.append(result["n_bad"])
            n_under.append(result["n_under"])
            n_over.append(result["n_over"])

        prefix = f"{scenario}_"
        output[prefix + "repop_ids"] = repop_ids.copy()
        output[prefix + "cdfs"] = np.asarray(cdfs, dtype=np.float64)
        output[prefix + "n_total"] = np.asarray(n_total, dtype=np.int64)
        output[prefix + "max_log10_js"] = np.asarray(
            max_log10_js,
            dtype=np.float64,
        )
        output[prefix + "n_bad"] = np.asarray(n_bad, dtype=np.int64)
        output[prefix + "n_under"] = np.asarray(n_under, dtype=np.int64)
        output[prefix + "n_over"] = np.asarray(n_over, dtype=np.int64)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **output)
    print(f"\nSaved: {args.output_npz}")


if __name__ == "__main__":
    main()
