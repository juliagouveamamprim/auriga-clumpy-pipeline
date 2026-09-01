#!/usr/bin/env python3
"""Check the distant-source approximation for repopulated subhalo catalogues.

For each input HDF5 catalogue, this diagnostic computes

    eta = r_s / D_Earth

for every subhalo, counts objects above configurable eta thresholds, and writes
individual rows only for objects above the 1% correction threshold.

The script writes CSV outputs and prints progress to stdout. Long-run logs should
be captured externally, e.g. with `tee`.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import h5py
import numpy as np


DEFAULT_ETA_1PCT = 0.445115210710
DEFAULT_ETA_5PCT = 0.879658561118


def infer_repop(path: str | Path) -> str:
    match = re.search(r"repop_(\d+)", str(path))
    return match.group(1) if match else ""


def infer_scenario(path: str | Path) -> str:
    text = str(path).lower()
    if "fragile" in text:
        return "fragile"
    if "resilient" in text:
        return "resilient"
    return ""


def decode_column_names(raw_names) -> list[str]:
    return [
        item.decode() if isinstance(item, bytes) else str(item)
        for item in raw_names
    ]


def find_data_dataset(h5: h5py.File) -> tuple[str, h5py.Dataset]:
    if "iteration_0/data" in h5:
        return "iteration_0/data", h5["iteration_0/data"]

    datasets: list[tuple[str, h5py.Dataset]] = []

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            datasets.append((name, obj))

    h5.visititems(collect)

    if len(datasets) == 1:
        return datasets[0]

    names = [name for name, _ in datasets]
    raise RuntimeError(f"Could not identify data dataset. Found: {names}")


def correction_ratio_nfw(eta: np.ndarray, nquad: int = 160) -> np.ndarray:
    """Return J_exact / J_approx for truncated NFW, valid for 0 < eta < 1."""
    eta = np.asarray(eta, dtype=float)
    ratio = np.full_like(eta, np.nan, dtype=float)

    valid = (eta > 0.0) & (eta < 1.0)
    if not np.any(valid):
        return ratio

    x, w = np.polynomial.legendre.leggauss(nquad)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w

    e = eta[valid][:, None]
    xx = x[None, :]

    log_term = np.log1p(e * xx) - np.log1p(-e * xx)
    integrand = log_term / (xx * (1.0 + xx) ** 4)
    integral = integrand @ w

    ratio[valid] = 12.0 * integral / (7.0 * eta[valid])
    return ratio


def write_header_if_needed(path: str | Path, fields: list[str]) -> None:
    path = Path(path)
    if not path.exists() or os.path.getsize(path) == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()


def already_processed(summary_path: str | Path) -> set[str]:
    path = Path(summary_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()

    with path.open() as handle:
        return {row["input_file"] for row in csv.DictReader(handle)}


def process_catalogue(path: str | Path, args) -> tuple[dict, list[dict]]:
    path = Path(path)
    outliers: list[dict] = []
    correction_values: list[float] = []

    with h5py.File(path, "r") as h5:
        dataset_name, data = find_data_dataset(h5)
        column_names = decode_column_names(data.attrs["column_names"])

        idx_distance = column_names.index("D_Earth")
        idx_rs = column_names.index("r_s")

        n_total = int(data.shape[0])
        n_valid = 0
        n_invalid = 0
        n_eta_gt_1pct = 0
        n_eta_gt_5pct = 0
        n_eta_ge_1 = 0
        max_eta = np.nan

        for start in range(0, n_total, args.chunksize):
            stop = min(start + args.chunksize, n_total)

            if start == 0 or start % args.progress_every == 0:
                percent = 100.0 * start / n_total
                print(
                    f"  processed {start:,}/{n_total:,} rows ({percent:.1f}%)",
                    flush=True,
                )

            distance = data[start:stop, idx_distance].astype(float)
            rs = data[start:stop, idx_rs].astype(float)

            valid = np.isfinite(distance) & np.isfinite(rs) & (distance > 0.0)
            eta = np.full(distance.shape, np.nan, dtype=float)
            eta[valid] = rs[valid] / distance[valid]

            above_1pct = valid & (eta > args.eta_1pct)
            above_5pct = valid & (eta > args.eta_5pct)
            above_or_equal_1 = valid & (eta >= 1.0)

            n_valid += int(np.count_nonzero(valid))
            n_invalid += int(np.count_nonzero(~valid))
            n_eta_gt_1pct += int(np.count_nonzero(above_1pct))
            n_eta_gt_5pct += int(np.count_nonzero(above_5pct))
            n_eta_ge_1 += int(np.count_nonzero(above_or_equal_1))

            if np.any(valid):
                max_eta = np.nanmax([max_eta, np.max(eta[valid])])

            if np.any(above_1pct):
                local_indices = np.where(above_1pct)[0]
                ratios = correction_ratio_nfw(eta[local_indices])
                corrections = (ratios - 1.0) * 100.0
                correction_values.extend(corrections[np.isfinite(corrections)])

                for local_index, ratio, correction in zip(
                    local_indices, ratios, corrections
                ):
                    outliers.append(
                        {
                            "repop": infer_repop(path),
                            "scenario": infer_scenario(path),
                            "input_file": str(path),
                            "dataset": dataset_name,
                            "row_index": start + int(local_index),
                            "eta": eta[local_index],
                            "correction_ratio": ratio,
                            "correction_percent": correction,
                            "leading_correction_percent": (
                                eta[local_index] ** 2 / 21.0 * 100.0
                            ),
                        }
                    )

        print(
            f"  processed {n_total:,}/{n_total:,} rows (100.0%)",
            flush=True,
        )

    summary = {
        "repop": infer_repop(path),
        "scenario": infer_scenario(path),
        "input_file": str(path),
        "dataset": dataset_name,
        "eta_1pct_threshold": args.eta_1pct,
        "eta_5pct_threshold": args.eta_5pct,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_eta_gt_1pct": n_eta_gt_1pct,
        "n_eta_gt_5pct": n_eta_gt_5pct,
        "n_eta_ge_1": n_eta_ge_1,
        "frac_eta_gt_1pct": n_eta_gt_1pct / n_total if n_total else np.nan,
        "frac_eta_gt_5pct": n_eta_gt_5pct / n_total if n_total else np.nan,
        "max_eta": max_eta,
        "correction_percent_min_eta_gt_1pct": (
            np.nanmin(correction_values) if correction_values else np.nan
        ),
        "correction_percent_median_eta_gt_1pct": (
            np.nanmedian(correction_values) if correction_values else np.nan
        ),
        "correction_percent_max_eta_gt_1pct": (
            np.nanmax(correction_values) if correction_values else np.nan
        ),
    }

    return summary, outliers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Input HDF5 catalogue files.")
    parser.add_argument("--file-list", help="Text file with one HDF5 path per line.")
    parser.add_argument("--summary", required=True, help="Output summary CSV.")
    parser.add_argument("--outliers", required=True, help="Output outlier CSV.")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--progress-every", type=int, default=10_000_000)
    parser.add_argument("--eta-1pct", type=float, default=DEFAULT_ETA_1PCT)
    parser.add_argument("--eta-5pct", type=float, default=DEFAULT_ETA_5PCT)
    args = parser.parse_args()

    files = list(args.files)
    if args.file_list:
        with open(args.file_list) as handle:
            files.extend(line.strip() for line in handle if line.strip())

    if not files:
        parser.error("Provide input files directly or with --file-list.")

    summary_fields = [
        "repop", "scenario", "input_file", "dataset",
        "eta_1pct_threshold", "eta_5pct_threshold",
        "n_total", "n_valid", "n_invalid",
        "n_eta_gt_1pct", "n_eta_gt_5pct", "n_eta_ge_1",
        "frac_eta_gt_1pct", "frac_eta_gt_5pct",
        "max_eta",
        "correction_percent_min_eta_gt_1pct",
        "correction_percent_median_eta_gt_1pct",
        "correction_percent_max_eta_gt_1pct",
    ]

    outlier_fields = [
        "repop", "scenario", "input_file", "dataset", "row_index",
        "eta", "correction_ratio", "correction_percent",
        "leading_correction_percent",
    ]

    write_header_if_needed(args.summary, summary_fields)
    write_header_if_needed(args.outliers, outlier_fields)

    done = already_processed(args.summary)

    print(f"Files requested: {len(files)}")
    print(f"Already processed: {len(done)}")

    for index, filename in enumerate(files, start=1):
        if filename in done:
            print(f"[{index:04d}/{len(files):04d}] SKIP {filename}", flush=True)
            continue

        print(f"[{index:04d}/{len(files):04d}] RUN {filename}", flush=True)
        summary, outliers = process_catalogue(filename, args)

        with open(args.summary, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=summary_fields).writerow(summary)

        if outliers:
            with open(args.outliers, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=outlier_fields)
                writer.writerows(outliers)

        print(
            f"[{index:04d}/{len(files):04d}] DONE "
            f"n_gt_1pct={summary['n_eta_gt_1pct']} "
            f"n_gt_5pct={summary['n_eta_gt_5pct']} "
            f"max_eta={summary['max_eta']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
