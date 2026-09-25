#!/usr/bin/env python3

"""Plot annular subhalo-to-smooth ratios from a previously saved CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = ("fragile", "resilient")
COLORS = {
    "fragile": "#EE3377",
    "resilient": "#009988",
}
REQUIRED_COLUMNS = {
    "scenario",
    "repop_id",
    "psi_min_deg",
    "psi_max_deg",
    "psi_center_deg",
    "n_pixels",
    "sum_j_smooth",
    "sum_j_pointlike",
    "sum_j_extended",
    "sum_j_sub",
    "ratio_sub_to_smooth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create the fragile/resilient two-panel annular "
            "J_sub/J_smooth figure from a saved CSV; no FITS are read."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--basename",
        default=None,
        help="Output basename without extension (default: CSV stem).",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats (default: png,pdf).",
    )
    return parser.parse_args()


def parse_formats(text):
    formats = tuple(item.strip().lower() for item in text.split(",") if item.strip())
    invalid = sorted(set(formats) - {"png", "pdf"})
    if not formats:
        raise ValueError("At least one output format is required.")
    if invalid:
        raise ValueError("Unsupported format(s): " + ", ".join(invalid))
    return formats


def load_records(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    grouped = {scenario: [] for scenario in SCENARIOS}
    repop_ids = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Input CSV is missing column(s): " + ", ".join(sorted(missing))
            )
        for row in reader:
            scenario = row["scenario"]
            if scenario not in grouped:
                raise ValueError(f"Unexpected scenario {scenario!r} in CSV.")
            repop_id = int(row["repop_id"])
            repop_ids.add(repop_id)
            record = {
                "repop_id": repop_id,
                "psi_min_deg": float(row["psi_min_deg"]),
                "psi_max_deg": float(row["psi_max_deg"]),
                "psi_center_deg": float(row["psi_center_deg"]),
                "n_pixels": int(row["n_pixels"]),
                "sum_j_smooth": float(row["sum_j_smooth"]),
                "sum_j_pointlike": float(row["sum_j_pointlike"]),
                "sum_j_extended": float(row["sum_j_extended"]),
                "sum_j_sub": float(row["sum_j_sub"]),
                "ratio_sub_to_smooth": float(row["ratio_sub_to_smooth"]),
            }
            grouped[scenario].append(record)

    if len(repop_ids) != 1:
        raise ValueError("CSV must contain exactly one repop_id.")
    if any(not grouped[scenario] for scenario in SCENARIOS):
        raise ValueError("CSV must contain both fragile and resilient rows.")

    for scenario, records in grouped.items():
        records.sort(key=lambda item: item["psi_min_deg"])
        ratios = np.asarray(
            [item["ratio_sub_to_smooth"] for item in records],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(ratios)) or np.any(ratios < 0.0):
            raise ValueError(
                f"{scenario}: ratios must be finite and non-negative."
            )
        for record in records:
            sums = np.asarray(
                [
                    record["sum_j_smooth"],
                    record["sum_j_pointlike"],
                    record["sum_j_extended"],
                    record["sum_j_sub"],
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(sums)) or np.any(sums < 0.0):
                raise ValueError(
                    f"{scenario}: CSV J-factor sums must be finite and "
                    "non-negative."
                )
            if record["sum_j_smooth"] <= 0.0:
                raise ValueError(
                    f"{scenario}: CSV smooth denominators must be positive."
                )
            expected_sub = record["sum_j_pointlike"] + record["sum_j_extended"]
            if not np.isclose(
                record["sum_j_sub"],
                expected_sub,
                rtol=1e-10,
                atol=0.0,
            ):
                raise ValueError(
                    f"{scenario}: CSV contains inconsistent subhalo sums."
                )
            expected_ratio = record["sum_j_sub"] / record["sum_j_smooth"]
            if not np.isclose(
                record["ratio_sub_to_smooth"],
                expected_ratio,
                rtol=1e-10,
                atol=0.0,
            ):
                raise ValueError(f"{scenario}: CSV contains inconsistent ratio.")

    fragile_edges = [
        (item["psi_min_deg"], item["psi_max_deg"])
        for item in grouped["fragile"]
    ]
    resilient_edges = [
        (item["psi_min_deg"], item["psi_max_deg"])
        for item in grouped["resilient"]
    ]
    if fragile_edges != resilient_edges:
        raise ValueError("Fragile and resilient CSV rows use different annuli.")

    return grouped, repop_ids.pop()


def shared_log_limits(grouped):
    ratios = np.concatenate(
        [
            np.asarray(
                [row["ratio_sub_to_smooth"] for row in grouped[scenario]],
                dtype=np.float64,
            )
            for scenario in SCENARIOS
        ]
    )
    values = np.concatenate(
        (ratios[ratios > 0.0], np.asarray([0.01, 0.1, 1.0]))
    )
    lower = 10.0 ** np.floor(np.log10(values.min()) - 0.15)
    upper = 10.0 ** np.ceil(np.log10(values.max()) + 0.15)
    return lower, upper


def make_figure(grouped, repop_id):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True, sharey=True)
    y_limits = shared_log_limits(grouped)

    for axis, scenario in zip(axes, SCENARIOS):
        records = grouped[scenario]
        x = np.asarray([row["psi_center_deg"] for row in records])
        y = np.asarray([row["ratio_sub_to_smooth"] for row in records])
        y = np.where(y > 0.0, y, np.nan)
        axis.plot(
            x,
            y,
            color=COLORS[scenario],
            marker="o",
            markersize=2.4,
            linewidth=1.35,
        )
        for reference in (0.01, 0.1, 1.0):
            axis.axhline(
                reference,
                color="0.55",
                linestyle="--",
                linewidth=0.8,
                alpha=0.75,
                zorder=0,
            )
        axis.set_title(f"{scenario.capitalize()} scenario")
        axis.set_xlim(0.0, 180.0)
        axis.set_ylim(*y_limits)
        axis.set_yscale("log")
        axis.grid(False, which="both")
        axis.tick_params(direction="in", which="both", top=True, right=True)

    axes[0].set_ylabel("Annular subhalo-to-smooth J-factor ratio")
    figure.supxlabel(
        r"Angular distance from the Galactic centre, $\psi_{\rm GC}$ [deg]",
        fontsize=14,
    )
    figure.suptitle(f"Repopulation {repop_id:04d}", fontsize=14)
    figure.tight_layout(w_pad=2.5)
    return figure


def main():
    args = parse_args()
    formats = parse_formats(args.formats)
    grouped, repop_id = load_records(args.input_csv)
    figure = make_figure(grouped, repop_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.basename or args.input_csv.stem
    if Path(basename).name != basename:
        raise ValueError("basename must not contain a directory component.")

    for output_format in formats:
        output_path = args.output_dir / f"{basename}.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = 220
        figure.savefig(output_path, **save_kwargs)
        print(f"Saved: {output_path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
