#!/usr/bin/env python3
"""Run and aggregate cut diagnostics across multiple repopulations."""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import healpy as hp
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SINGLE_SCAN_SCRIPT = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "cuts"
    / "scripts"
    / "scan_central_pixel_cuts.py"
)

DEFAULT_RESULTS_DIR = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "cuts"
    / "results_summary"
)

DEFAULT_LOGS_DIR = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "cuts"
    / "logs"
)

VALID_SCENARIOS = {"fragile", "resilient"}
VALID_ENVELOPE_MODES = {"theta-s"}

BASE_AGGREGATE_METRICS = [
    "n_valid_total",
    "n_pointlike_total",
    "n_pointlike_kept",
    "n_pointlike_discarded",
    "pointlike_reduction_factor",
    "n_extended_total",
    "n_extended_kept",
    "n_extended_discarded",
    "extended_reduction_factor",
    "n_total_kept",
    "total_reduction_factor",
    "fraction_sum_discarded_to_full",
    "ratio_max_discarded_to_final",
]


def parse_float_values(text):
    values = sorted(
        {
            float(value.strip())
            for value in text.split(",")
            if value.strip()
        }
    )

    if not values:
        raise ValueError("At least one cut fraction must be provided.")

    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Cut fractions must be finite and non-negative.")

    return values


def parse_scenarios(text):
    scenarios = list(
        dict.fromkeys(
            value.strip().lower()
            for value in text.split(",")
            if value.strip()
        )
    )

    invalid = sorted(set(scenarios) - VALID_SCENARIOS)

    if invalid:
        raise ValueError(
            "Invalid scenario(s): " + ", ".join(invalid)
        )

    if not scenarios:
        raise ValueError("At least one scenario must be provided.")

    return scenarios


def parse_envelope_modes(text):
    normalized = text.strip().lower()

    if normalized in {"", "none"}:
        return []

    modes = list(
        dict.fromkeys(
            value.strip()
            for value in normalized.split(",")
            if value.strip()
        )
    )

    invalid = sorted(set(modes) - VALID_ENVELOPE_MODES)

    if invalid:
        raise ValueError(
            "Invalid envelope mode(s): " + ", ".join(invalid)
        )

    return modes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the catalogue-wise cut diagnostic over multiple "
            "repopulations and aggregate the resulting CSV files."
        )
    )

    parser.add_argument(
        "--repop-start",
        type=int,
        default=0,
        help="First repopulation ID.",
    )

    parser.add_argument(
        "--n-repops",
        type=int,
        default=10,
        help="Number of consecutive repopulations.",
    )

    parser.add_argument(
        "--scenarios",
        default="fragile,resilient",
        help="Comma-separated scenarios.",
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing repop_XXXX directories. "
            "When omitted, the single-scan default path is used."
        ),
    )

    parser.add_argument(
        "--nside",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--pointlike-f-values",
        default="1e-2,1e-3,1e-4,1e-5",
    )

    parser.add_argument(
        "--extended-f-values",
        default="1e-2,1e-3,1e-4,1e-5",
    )

    parser.add_argument(
        "--theta-aperture-deg",
        type=float,
        default=None,
        help=(
            "Deprecated. The extended proxy now uses the CLUMPY "
            "aperture hp.max_pixrad(NSIDE), derived automatically."
        ),
    )

    parser.add_argument(
        "--extended-envelope-modes",
        default="theta-s",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
    )

    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing individual scans.",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not run scans; aggregate existing CSV files only.",
    )

    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing inputs or individual CSV files.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after an individual scan fails.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    return parser.parse_args()


def individual_stem(repop_id, scenario, nside, envelope_modes):
    suffix = (
        "envelopes_scan"
        if envelope_modes
        else "central_pixel_scan"
    )

    return (
        f"repop_{repop_id:04d}_{scenario}"
        f"_nside{nside}_{suffix}"
    )


def build_input_h5(input_root, repop_id, scenario):
    if input_root is None:
        return None

    return (
        input_root
        / f"repop_{repop_id:04d}"
        / f"fullrepop_hydro_{scenario}.h5"
    )


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows available for {path}")

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def float_set(rows, column):
    return {
        float(row[column])
        for row in rows
    }


def validate_existing_csv(
    path,
    repop_id,
    scenario,
    nside,
    pointlike_f_values,
    extended_f_values,
    envelope_modes,
):
    rows = read_csv_rows(path)

    if not rows:
        raise ValueError(f"Empty result CSV: {path}")

    expected_modes = (
        ",".join(envelope_modes)
        if envelope_modes
        else "none"
    )

    checks = {
        "repop_id": str(repop_id),
        "scenario": scenario,
        "nside": str(nside),
        "extended_envelope_modes": expected_modes,
    }

    for column, expected in checks.items():
        observed = {row[column] for row in rows}

        if observed != {expected}:
            raise ValueError(
                f"Existing CSV has incompatible {column}: "
                f"{path} -> {sorted(observed)}"
            )

    if float_set(rows, "pointlike_f") != set(pointlike_f_values):
        raise ValueError(
            f"Existing CSV has incompatible pointlike cuts: {path}"
        )

    if float_set(rows, "extended_f") != set(extended_f_values):
        raise ValueError(
            f"Existing CSV has incompatible extended cuts: {path}"
        )

    expected_alpha_int_deg = float(
        np.rad2deg(hp.max_pixrad(nside))
    )

    observed = {
        float(row["theta_aperture_deg"])
        for row in rows
    }

    if not all(
        np.isclose(
            value,
            expected_alpha_int_deg,
            rtol=0.0,
            atol=1.0e-10,
        )
        for value in observed
    ):
        raise ValueError(
            "Existing CSV has an incompatible or legacy aperture: "
            f"{path}; expected "
            f"{expected_alpha_int_deg:.12f} deg, "
            f"found {sorted(observed)}"
        )

    return rows


def build_scan_command(
    args,
    repop_id,
    scenario,
    input_h5,
    output_csv,
):
    command = [
        sys.executable,
        "-u",
        str(SINGLE_SCAN_SCRIPT),
        str(repop_id),
        scenario,
        "--nside",
        str(args.nside),
        "--pointlike-f-values",
        args.pointlike_f_values,
        "--extended-f-values",
        args.extended_f_values,
        "--extended-envelope-modes",
        args.extended_envelope_modes,
        "--output-csv",
        str(output_csv),
    ]

    if input_h5 is not None:
        command.extend(["--input-h5", str(input_h5)])

    if args.chunk_size is not None:
        command.extend(
            ["--chunk-size", str(args.chunk_size)]
        )

    return command


def run_scan(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("Command:")
    print(" ".join(command))
    print()

    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()

        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
        )


def add_derived_columns(rows, source_csv):
    enriched = []

    for original in rows:
        row = dict(original)

        n_pointlike_kept = int(row["n_pointlike_kept"])
        n_extended_kept = int(row["n_extended_kept"])
        n_valid_total = int(row["n_valid_total"])

        n_total_kept = (
            n_pointlike_kept
            + n_extended_kept
        )

        total_reduction_factor = (
            n_valid_total / n_total_kept
            if n_total_kept > 0
            else float("inf")
        )

        row["n_total_kept"] = n_total_kept
        row["total_reduction_factor"] = (
            total_reduction_factor
        )
        row["source_csv"] = str(source_csv)

        enriched.append(row)

    return enriched


def aggregate_rows(rows):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["scenario"],
            int(row["nside"]),
            float(row["pointlike_f"]),
            float(row["extended_f"]),
            float(row["theta_aperture_deg"]),
            row["extended_envelope_modes"],
        )
        grouped[key].append(row)

    ratio_metrics = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("ratio_max_discarded")
            and key.endswith("_to_final")
        }
    )

    metrics = list(BASE_AGGREGATE_METRICS)

    for metric in ratio_metrics:
        if metric not in metrics:
            metrics.append(metric)

    summary_rows = []

    for key, group_rows in sorted(grouped.items()):
        (
            scenario,
            nside,
            pointlike_f,
            extended_f,
            theta_aperture_deg,
            envelope_modes,
        ) = key

        repop_ids = sorted(
            {
                int(row["repop_id"])
                for row in group_rows
            }
        )

        summary = {
            "scenario": scenario,
            "nside": nside,
            "pointlike_f": pointlike_f,
            "extended_f": extended_f,
            "theta_aperture_deg": theta_aperture_deg,
            "extended_envelope_modes": envelope_modes,
            "n_repops": len(repop_ids),
            "repop_id_min": min(repop_ids),
            "repop_id_max": max(repop_ids),
        }

        for metric in metrics:
            values_and_repops = []

            for row in group_rows:
                if metric not in row or row[metric] == "":
                    continue

                value = float(row[metric])

                if np.isfinite(value):
                    values_and_repops.append(
                        (value, int(row["repop_id"]))
                    )

            if not values_and_repops:
                continue

            values = np.asarray(
                [item[0] for item in values_and_repops],
                dtype=float,
            )

            summary[f"{metric}_mean"] = float(
                np.mean(values)
            )
            summary[f"{metric}_std"] = float(
                np.std(values, ddof=1)
                if len(values) > 1
                else 0.0
            )
            summary[f"{metric}_median"] = float(
                np.median(values)
            )
            summary[f"{metric}_min"] = float(
                np.min(values)
            )
            summary[f"{metric}_max"] = float(
                np.max(values)
            )

            if metric in ratio_metrics:
                maximum = max(
                    values_and_repops,
                    key=lambda item: item[0],
                )
                summary[
                    f"{metric}_max_repop_id"
                ] = maximum[1]

        summary_rows.append(summary)

    return summary_rows


def print_summary(summary_rows):
    print()
    print("=" * 80)
    print("Multi-repopulation summary")
    print("=" * 80)

    for row in summary_rows:
        print(
            f"{row['scenario']:9s} | "
            f"f_pl={float(row['pointlike_f']):.1e} "
            f"f_ext={float(row['extended_f']):.1e} | "
            f"N={row['n_repops']}"
        )

        ratio_columns = sorted(
            key
            for key in row
            if key.startswith("ratio_max_discarded")
            and key.endswith("_mean")
        )

        for mean_key in ratio_columns:
            metric = mean_key.removesuffix("_mean")

            print(
                f"  {metric}: "
                f"mean={100.0 * float(row[mean_key]):.4f}% "
                f"std={100.0 * float(row[metric + '_std']):.4f}% "
                f"min={100.0 * float(row[metric + '_min']):.4f}% "
                f"max={100.0 * float(row[metric + '_max']):.4f}% "
                f"(repop "
                f"{int(row[metric + '_max_repop_id']):04d})"
            )

        print(
            "  retained: "
            f"pointlike mean="
            f"{float(row['n_pointlike_kept_mean']):,.0f}, "
            f"extended mean="
            f"{float(row['n_extended_kept_mean']):,.0f}, "
            f"total mean="
            f"{float(row['n_total_kept_mean']):,.0f}"
        )

        print(
            "  total reduction: "
            f"mean=×"
            f"{float(row['total_reduction_factor_mean']):.1f}, "
            f"range=×"
            f"{float(row['total_reduction_factor_min']):.1f}"
            "–×"
            f"{float(row['total_reduction_factor_max']):.1f}"
        )

    print("=" * 80)


def main():
    args = parse_args()

    if args.theta_aperture_deg is not None:
        raise ValueError(
            "--theta-aperture-deg is deprecated. The aperture is now "
            "derived automatically as hp.max_pixrad(NSIDE)."
        )

    if args.repop_start < 0:
        raise ValueError("repop-start must be non-negative.")

    if args.n_repops <= 0:
        raise ValueError("n-repops must be positive.")

    scenarios = parse_scenarios(args.scenarios)
    pointlike_f_values = parse_float_values(
        args.pointlike_f_values
    )
    extended_f_values = parse_float_values(
        args.extended_f_values
    )
    envelope_modes = parse_envelope_modes(
        args.extended_envelope_modes
    )

    repop_ids = range(
        args.repop_start,
        args.repop_start + args.n_repops,
    )

    repop_end = (
        args.repop_start
        + args.n_repops
        - 1
    )

    scenario_tag = "-".join(scenarios)

    prefix = (
        f"repops_{args.repop_start:04d}_{repop_end:04d}_"
        f"{scenario_tag}_nside{args.nside}"
    )

    combined_csv = (
        args.combined_csv
        if args.combined_csv is not None
        else args.results_dir / f"{prefix}_combined.csv"
    )

    summary_csv = (
        args.summary_csv
        if args.summary_csv is not None
        else args.results_dir / f"{prefix}_summary.csv"
    )

    collected_rows = []
    failures = []

    for repop_id in repop_ids:
        for scenario in scenarios:
            stem = individual_stem(
                repop_id,
                scenario,
                args.nside,
                envelope_modes,
            )

            output_csv = (
                args.results_dir
                / f"{stem}.csv"
            )

            log_path = (
                args.logs_dir
                / f"{stem}.log"
            )

            input_h5 = build_input_h5(
                args.input_root,
                repop_id,
                scenario,
            )

            print()
            print("=" * 80)
            print(
                f"repop_{repop_id:04d} | {scenario}"
            )
            print("=" * 80)

            if (
                input_h5 is not None
                and not input_h5.exists()
            ):
                message = f"Missing input: {input_h5}"

                if args.skip_missing:
                    print(f"Skipping: {message}")
                    continue

                raise FileNotFoundError(message)

            try:
                if output_csv.exists() and not args.overwrite:
                    print(
                        f"Reusing existing result: {output_csv}"
                    )

                    rows = validate_existing_csv(
                        output_csv,
                        repop_id,
                        scenario,
                        args.nside,
                        pointlike_f_values,
                        extended_f_values,
                        envelope_modes,
                    )
                elif args.aggregate_only:
                    message = (
                        f"Missing individual result: {output_csv}"
                    )

                    if args.skip_missing:
                        print(f"Skipping: {message}")
                        continue

                    raise FileNotFoundError(message)
                else:
                    command = build_scan_command(
                        args,
                        repop_id,
                        scenario,
                        input_h5,
                        output_csv,
                    )

                    if args.dry_run:
                        print(" ".join(command))
                        continue

                    run_scan(command, log_path)

                    rows = validate_existing_csv(
                        output_csv,
                        repop_id,
                        scenario,
                        args.nside,
                        pointlike_f_values,
                        extended_f_values,
                        envelope_modes,
                    )

                collected_rows.extend(
                    add_derived_columns(
                        rows,
                        output_csv,
                    )
                )

            except Exception as error:
                failures.append(
                    (
                        repop_id,
                        scenario,
                        str(error),
                    )
                )

                if not args.continue_on_error:
                    raise

                print(
                    f"Failed repop_{repop_id:04d} "
                    f"{scenario}: {error}"
                )

    if args.dry_run:
        return

    if not collected_rows:
        raise RuntimeError(
            "No individual results were collected."
        )

    collected_rows.sort(
        key=lambda row: (
            row["scenario"],
            int(row["repop_id"]),
            float(row["pointlike_f"]),
            float(row["extended_f"]),
        )
    )

    write_csv_rows(
        combined_csv,
        collected_rows,
    )

    summary_rows = aggregate_rows(
        collected_rows
    )

    write_csv_rows(
        summary_csv,
        summary_rows,
    )

    print_summary(summary_rows)

    print()
    print(f"Combined CSV: {combined_csv}")
    print(f"Summary CSV:  {summary_csv}")

    if failures:
        print()
        print("Failures:")

        for repop_id, scenario, message in failures:
            print(
                f"  repop_{repop_id:04d} "
                f"{scenario}: {message}"
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
