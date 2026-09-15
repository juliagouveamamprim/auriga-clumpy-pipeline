#!/usr/bin/env python3
"""Plot mean and percentile distributions of discarded-subhalo relative Js."""

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
CLI_POPULATIONS = ("combined",) + POPULATIONS
SCENARIO_ORDER = ("fragile", "resilient")
DISPLAY_X_LIMITS = (1.0e-12, 1.0e-2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the arithmetic mean and pointwise 16th--84th percentiles "
            "of the normalized per-catalogue discarded-subhalo Js "
            "distributions."
        )
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--population",
        choices=CLI_POPULATIONS,
        default="combined",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats (default: png,pdf).",
    )
    parser.add_argument(
        "--rebin-factor",
        type=int,
        default=2,
        help="Number of adjacent histogram bins to sum (default: 2).",
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


def rebin_histograms(
    histograms: np.ndarray,
    log_relative_edges: np.ndarray,
    rebin_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum adjacent histogram bins and return their new log-space edges."""
    histograms = np.asarray(histograms)
    edges = np.asarray(log_relative_edges, dtype=np.float64)

    if histograms.ndim != 2:
        raise ValueError("Histograms must be a two-dimensional array.")
    if edges.ndim != 1 or histograms.shape[1] != len(edges) - 1:
        raise ValueError("Histogram bins do not match log-relative edges.")
    if (
        isinstance(rebin_factor, (bool, np.bool_))
        or not isinstance(rebin_factor, (int, np.integer))
        or rebin_factor <= 0
    ):
        raise ValueError("Rebin factor must be a positive integer.")

    n_bins = histograms.shape[1]
    if n_bins % rebin_factor != 0:
        raise ValueError(
            f"Original bin count ({n_bins}) must be divisible by "
            f"rebin factor ({rebin_factor})."
        )

    rebinned = histograms.reshape(
        histograms.shape[0], n_bins // rebin_factor, rebin_factor
    ).sum(axis=2)
    rebinned_edges = edges[::rebin_factor]
    return rebinned, rebinned_edges


def aggregate_repop_distributions(
    histograms: np.ndarray,
    totals: np.ndarray,
    log_relative_edges: np.ndarray,
    rebin_factor: int,
) -> dict[str, np.ndarray]:
    """Rebin, normalize each catalogue, then aggregate bin by bin."""
    histograms = np.asarray(histograms, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)

    if histograms.ndim != 2 or histograms.shape[0] != len(totals):
        raise ValueError("Histogram and total arrays are inconsistent.")
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError(
            "Each catalogue must have a finite, positive discarded total."
        )
    if np.any(~np.isfinite(histograms)) or np.any(histograms < 0.0):
        raise ValueError("Histogram counts must be finite and non-negative.")

    rebinned, rebinned_edges = rebin_histograms(
        histograms,
        log_relative_edges,
        rebin_factor,
    )
    percentage = 100.0 * rebinned / totals[:, None]
    return {
        "log_relative_edges": rebinned_edges,
        "percentage": percentage,
        "mean": np.mean(percentage, axis=0),
        "lower": np.percentile(percentage, 16.0, axis=0),
        "upper": np.percentile(percentage, 84.0, axis=0),
    }


def percentage_axis_label(log_relative_edges: np.ndarray) -> str:
    """Return the percentage label after validating uniform log edges."""
    widths = np.diff(np.asarray(log_relative_edges, dtype=np.float64))
    if len(widths) == 0 or not np.allclose(widths, widths[0]):
        raise ValueError("Rebinned log-relative bins must have uniform widths.")
    return r"Fraction of discarded subhalos [\%]"


def plot_distribution(
    data: dict[str, np.ndarray],
    population: str,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    rebin_factor: int,
) -> list[Path]:
    if population == "combined":
        return plot_combined_distribution(
            data=data,
            output_dir=output_dir,
            formats=formats,
            dpi=dpi,
            rebin_factor=rebin_factor,
        )

    return plot_single_population_distribution(
        data=data,
        population=population,
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
        rebin_factor=rebin_factor,
    )


def _scenario_order(scenarios: np.ndarray) -> list[str]:
    available = set(scenarios)
    ordered = [
        scenario for scenario in SCENARIO_ORDER if scenario in available
    ]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _draw_population(
    axis,
    data: dict[str, np.ndarray],
    population: str,
    rebin_factor: int,
    *,
    add_legend: bool,
) -> tuple[np.ndarray, list, list]:
    edges = np.asarray(data["log_relative_edges"], dtype=np.float64)
    scenarios = np.asarray(data["catalogue_scenarios"], dtype=str)
    histograms = np.asarray(data[f"hist_{population}"], dtype=np.float64)
    totals = np.asarray(data[f"n_discarded_{population}"], dtype=np.float64)
    colors = {
        "fragile": "#EE3377",
        "resilient": "#009988",
    }

    scenario_order = _scenario_order(scenarios)
    legend_handles = []
    legend_labels = []
    rebinned_log_edges = None

    for scenario in scenario_order:
        selected = scenarios == scenario
        aggregate = aggregate_repop_distributions(
            histograms[selected],
            totals[selected],
            edges,
            rebin_factor,
        )
        color = colors.get(scenario, "0.25")
        label = scenario.capitalize()
        rebinned_log_edges = aggregate["log_relative_edges"]
        rebinned_edges = 10.0 ** rebinned_log_edges
        band_handle = axis.stairs(
            aggregate["upper"],
            rebinned_edges,
            baseline=aggregate["lower"],
            color=color,
            alpha=0.18,
            linewidth=0.0,
            label=f"{label} bin-wise 16--84 percentile",
            fill=True,
            zorder=1,
        )
        mean_handle = axis.stairs(
            aggregate["mean"],
            rebinned_edges,
            color=color,
            linewidth=2.2,
            label=f"{label} mean",
            fill=False,
            zorder=2,
        )
        legend_handles.extend((mean_handle, band_handle))
        legend_labels.extend(
            (f"{label} mean", f"{label} bin-wise 16--84 percentile")
        )

    if rebinned_log_edges is None:
        raise ValueError("No catalogue scenarios are available to plot.")

    axis.set_xscale("log")
    axis.set_xlim(*DISPLAY_X_LIMITS)
    axis.tick_params(axis="both", which="both", labelsize=15)
    axis.xaxis.get_offset_text().set_fontsize(15)
    axis.yaxis.get_offset_text().set_fontsize(15)
    axis.grid(False, which="both")
    if add_legend:
        axis.legend(
            handles=legend_handles,
            labels=legend_labels,
            ncols=1,
            loc="upper right",
            fontsize=12,
            labelspacing=0.3,
            handlelength=2.0,
            handletextpad=0.5,
            borderaxespad=0.4,
            frameon=False,
        )

    return rebinned_log_edges, legend_handles, legend_labels


def _save_figure(
    figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:

    output_dir.mkdir(parents=True, exist_ok=True)
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


def plot_single_population_distribution(
    data: dict[str, np.ndarray],
    population: str,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    rebin_factor: int,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    rebinned_log_edges, _, _ = _draw_population(
        axis,
        data,
        population,
        rebin_factor,
        add_legend=True,
    )
    axis.set_xlabel(r"$J_s/J_{s,\max}^{\rm cat}$")
    axis.set_ylabel(percentage_axis_label(rebinned_log_edges), fontsize=15)
    return _save_figure(
        figure,
        output_dir,
        f"discarded_js_distribution_{population}",
        formats,
        dpi,
    )


def plot_combined_distribution(
    data: dict[str, np.ndarray],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    rebin_factor: int,
) -> list[Path]:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.5, 8.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_populations = ("pointlike", "extended")
    panel_titles = (
        "Pointlike subhalos",
        "Extended subhalos",
    )
    aggregates = []
    for axis, population, title in zip(axes, panel_populations, panel_titles):
        rebinned_log_edges, _, _ = _draw_population(
            axis,
            data,
            population,
            rebin_factor,
            add_legend=population == "pointlike",
        )
        axis.text(
            0.02,
            0.93,
            title,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=16,
        )
        axis.set_xlabel("")
        aggregates.append((rebinned_log_edges, axis))

    y_max = 0.0
    for population in panel_populations:
        scenarios = np.asarray(data["catalogue_scenarios"], dtype=str)
        histograms = np.asarray(data[f"hist_{population}"], dtype=np.float64)
        totals = np.asarray(data[f"n_discarded_{population}"], dtype=np.float64)
        for scenario in _scenario_order(scenarios):
            aggregate = aggregate_repop_distributions(
                histograms[scenarios == scenario],
                totals[scenarios == scenario],
                data["log_relative_edges"],
                rebin_factor,
            )
            y_max = max(y_max, float(np.max(aggregate["upper"])))

    axes[0].set_ylim(0.0, y_max * 1.05 if y_max > 0.0 else 1.0)
    axes[-1].set_xlabel(r"$J_s/J_{s,\max}^{\rm cat}$")
    figure.supylabel(
        percentage_axis_label(aggregates[0][0]),
        fontsize=15,
    )
    return _save_figure(
        figure,
        output_dir,
        "discarded_js_distribution_pointlike_extended",
        formats,
        dpi,
    )


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
        rebin_factor=args.rebin_factor,
    )


if __name__ == "__main__":
    main()
