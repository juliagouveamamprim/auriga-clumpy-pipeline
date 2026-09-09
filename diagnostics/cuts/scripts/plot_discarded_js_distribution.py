#!/usr/bin/env python3
"""Plot per-catalogue distributions of discarded-subhalo relative Js."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


FORMAT_VERSION = 1
POPULATIONS = ("all", "pointlike", "extended")
SCENARIO_ORDER = ("fragile", "resilient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the mean and 16th--84th percentile band of the "
            "per-catalogue discarded-subhalo Js distributions."
        )
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--population",
        choices=POPULATIONS,
        default="all",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats (default: png,pdf).",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_formats(text: str) -> tuple[str, ...]:
    formats = tuple(
        item.strip().lower()
        for item in text.split(",")
        if item.strip()
    )
    invalid = sorted(set(formats) - {"png", "pdf"})

    if not formats:
        raise ValueError("At least one output format is required.")
    if invalid:
        raise ValueError("Unsupported format(s): " + ", ".join(invalid))
    return formats


def load_distribution(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {
            "format_version",
            "log_relative_edges",
            "requested_repop_ids",
            "requested_scenarios",
            "catalogue_repop_ids",
            "catalogue_scenarios",
        }
        for population in POPULATIONS:
            required.add(f"hist_{population}")
            required.add(f"n_discarded_{population}")

        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Input NPZ is missing fields: {missing}")
        if int(data["format_version"]) != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format version: {int(data['format_version'])}"
            )

        output = {
            name: np.asarray(data[name])
            for name in required
        }

    edges = np.asarray(output["log_relative_edges"], dtype=np.float64)
    repop_ids = np.asarray(output["catalogue_repop_ids"], dtype=np.int64)
    scenarios = np.asarray(output["catalogue_scenarios"], dtype=str)
    requested_repop_ids = np.asarray(
        output["requested_repop_ids"], dtype=np.int64
    )
    requested_scenarios = np.asarray(
        output["requested_scenarios"], dtype=str
    )
    n_catalogues = len(repop_ids)

    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("log_relative_edges must be strictly increasing.")
    if len(scenarios) != n_catalogues:
        raise ValueError("Catalogue scenario and repop arrays are inconsistent.")
    catalogue_keys = set(zip(scenarios, repop_ids))
    if len(catalogue_keys) != n_catalogues:
        raise ValueError("Duplicate catalogue keys in input NPZ.")
    requested_keys = {
        (scenario, repop_id)
        for scenario in requested_scenarios
        for repop_id in requested_repop_ids
    }
    if catalogue_keys != requested_keys:
        raise ValueError(
            "Input NPZ is an incomplete or incompatible checkpoint."
        )

    for population in POPULATIONS:
        histogram = output[f"hist_{population}"]
        totals = output[f"n_discarded_{population}"]
        if histogram.shape != (n_catalogues, len(edges) - 1):
            raise ValueError(
                f"Invalid hist_{population} shape: {histogram.shape}"
            )
        if totals.shape != (n_catalogues,):
            raise ValueError(
                f"Invalid n_discarded_{population} shape: {totals.shape}"
            )

    return output


def aggregate_repop_distributions(
    histograms: np.ndarray,
    totals: np.ndarray,
    log_relative_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Normalize each catalogue first, then aggregate bin by bin."""
    histograms = np.asarray(histograms, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    widths = np.diff(np.asarray(log_relative_edges, dtype=np.float64))

    if histograms.ndim != 2 or histograms.shape[0] != len(totals):
        raise ValueError("Histogram and total arrays are inconsistent.")
    if histograms.shape[1] != len(widths):
        raise ValueError("Histogram bins do not match log-relative edges.")
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError(
            "Each catalogue must have a finite, positive discarded total."
        )
    if np.any(~np.isfinite(histograms)) or np.any(histograms < 0.0):
        raise ValueError("Histogram counts must be finite and non-negative.")

    density = histograms / totals[:, None] / widths[None, :]
    return {
        "density": density,
        "mean": np.mean(density, axis=0),
        "lower": np.percentile(density, 16.0, axis=0),
        "upper": np.percentile(density, 84.0, axis=0),
    }


def plot_distribution(
    data: dict[str, np.ndarray],
    population: str,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    edges = np.asarray(data["log_relative_edges"], dtype=np.float64)
    centers = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
    scenarios = np.asarray(data["catalogue_scenarios"], dtype=str)
    histograms = np.asarray(data[f"hist_{population}"], dtype=np.float64)
    totals = np.asarray(data[f"n_discarded_{population}"], dtype=np.float64)
    colors = {
        "fragile": "#EE3377",
        "resilient": "#009988",
    }

    figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    available = set(scenarios)
    scenario_order = [
        scenario for scenario in SCENARIO_ORDER if scenario in available
    ]
    scenario_order.extend(sorted(available - set(scenario_order)))

    for scenario in scenario_order:
        selected = scenarios == scenario
        aggregate = aggregate_repop_distributions(
            histograms[selected],
            totals[selected],
            edges,
        )
        color = colors.get(scenario, "0.25")
        label = scenario.capitalize()
        axis.fill_between(
            centers,
            aggregate["lower"],
            aggregate["upper"],
            color=color,
            alpha=0.20,
            linewidth=0.0,
            label=f"{label} 16--84 percentile",
        )
        axis.plot(
            centers,
            aggregate["mean"],
            color=color,
            linewidth=2.2,
            label=f"{label} mean",
        )

    axis.set_xscale("log")
    axis.set_xlim(10.0**edges[0], 10.0**edges[-1])
    axis.set_xlabel(r"$J_s/J_{s,\max}^{\rm cat}$")
    axis.set_ylabel(
        "Fraction of discarded subhalos\n"
        r"per unit $\log_{10}(J_s/J_{s,\max}^{\rm cat})$"
    )
    axis.tick_params(axis="both", which="both", labelsize=15)
    axis.xaxis.get_offset_text().set_fontsize(15)
    axis.yaxis.get_offset_text().set_fontsize(15)
    axis.grid(False, which="both")
    axis.legend(frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"discarded_js_distribution_{population}"
    saved = []

    for output_format in formats:
        output_path = output_dir / f"{stem}.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = dpi
        figure.savefig(output_path, **save_kwargs)
        saved.append(output_path)
        print(f"Saved: {output_path}")

    plt.close(figure)
    return saved


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)
    data = load_distribution(args.input_npz)
    plot_distribution(
        data=data,
        population=args.population,
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
