#!/usr/bin/env python3
"""Plot multi-repopulation catalog-cut diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullLocator


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15.5,
    }
)


REQUIRED_COLUMNS = {
    "scenario",
    "repop_id",
    "nside",
    "pointlike_f",
    "extended_f",
    "ratio_max_discarded_theta_s_envelope_to_final",
    "fraction_discarded_js_to_full",
    "n_pointlike_total",
    "n_pointlike_kept",
    "n_pointlike_discarded",
    "n_extended_total",
    "n_extended_kept",
    "n_extended_discarded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot conservative cut impact and catalog retention "
            "for the diagonal pointlike/extended cut scan."
        )
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        required=True,
        help="Combined per-repopulation CSV from scan_multiple_repops.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which the figures will be saved.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats. Default: png,pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG resolution. Default: 220.",
    )
    return parser.parse_args()


def validate_columns(table: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise ValueError(
            "The combined CSV is missing required columns: "
            + ", ".join(missing)
        )


def compact_count(value: float) -> str:
    """Format a catalog count compactly."""
    value = float(value)
    magnitude = abs(value)

    if magnitude >= 1.0e9:
        return f"{value / 1.0e9:.2f}B"
    if magnitude >= 1.0e6:
        return f"{value / 1.0e6:.2f}M"
    if magnitude >= 1.0e3:
        return f"{value / 1.0e3:.1f}k"
    return f"{value:.0f}"


def percent_tick(value: float, _position: float) -> str:
    return rf"{value:g}\%"


def prepare_diagonal(table: pd.DataFrame) -> pd.DataFrame:
    diagonal = table[
        np.isclose(
            table["pointlike_f"].to_numpy(dtype=float),
            table["extended_f"].to_numpy(dtype=float),
        )
    ].copy()

    if diagonal.empty:
        raise ValueError(
            "No rows with pointlike_f == extended_f were found."
        )

    diagonal["cut_f"] = diagonal["pointlike_f"].astype(float)
    diagonal["impact_percent"] = (
        100.0
        * diagonal[
            "ratio_max_discarded_theta_s_envelope_to_final"
        ].astype(float)
    )
    diagonal["integrated_discarded_percent"] = (
        100.0
        * diagonal["fraction_discarded_js_to_full"].astype(float)
    )

    for population in ("pointlike", "extended"):
        total = diagonal[f"n_{population}_total"].astype(float)
        kept = diagonal[f"n_{population}_kept"].astype(float)

        if (total <= 0).any():
            raise ValueError(
                f"Non-positive {population} catalog size found."
            )

        diagonal[f"{population}_kept_percent"] = 100.0 * kept / total

    return diagonal


def aggregate_retention(
    scenario_table: pd.DataFrame,
    f_values: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = {
        "pointlike_kept_percent": [],
        "pointlike_kept_count": [],
        "pointlike_discarded_count": [],
        "extended_kept_percent": [],
        "extended_kept_count": [],
        "extended_discarded_count": [],
    }

    for cut_f in f_values:
        rows = scenario_table[
            np.isclose(
                scenario_table["cut_f"].to_numpy(dtype=float),
                cut_f,
            )
        ]

        for population in ("pointlike", "extended"):
            result[f"{population}_kept_percent"].append(
                rows[f"{population}_kept_percent"].mean()
            )
            result[f"{population}_kept_count"].append(
                rows[f"n_{population}_kept"].mean()
            )
            result[f"{population}_discarded_count"].append(
                rows[f"n_{population}_discarded"].mean()
            )

    return {
        key: np.asarray(values, dtype=float)
        for key, values in result.items()
    }


def metric_matrix(
    scenario_table: pd.DataFrame,
    f_values: np.ndarray,
    metric: str,
) -> np.ndarray:
    matrix = scenario_table.pivot(
        index="repop_id",
        columns="cut_f",
        values=metric,
    ).reindex(columns=f_values)

    if matrix.isna().any().any():
        raise ValueError(
            f"Incomplete diagonal scan for metric {metric!r}."
        )

    return matrix.to_numpy(dtype=float)


def plot_metric_distribution(
    axis,
    x: np.ndarray,
    values: np.ndarray,
    color: str,
    label: str,
) -> None:
    lower = np.percentile(values, 16.0, axis=0)
    mean = np.mean(values, axis=0)
    upper = np.percentile(values, 84.0, axis=0)

    for repop_values in values:
        axis.plot(
            x,
            repop_values,
            color=color,
            linewidth=0.8,
            alpha=0.18,
            zorder=1,
        )

    axis.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=0.18,
        linewidth=0.0,
        zorder=2,
    )
    axis.plot(
        x,
        mean,
        color=color,
        marker="o",
        linewidth=2.2,
        label=label,
        zorder=3,
    )


def plot_combined(
    diagonal: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    scenario_colors = {
        "fragile": "#EE3377",
        "resilient": "#009988",
    }
    scenario_order = ("fragile", "resilient")

    available = set(diagonal["scenario"].astype(str).unique())
    scenarios = [
        scenario
        for scenario in scenario_order
        if scenario in available
    ]
    scenarios.extend(sorted(available.difference(scenarios)))

    if not scenarios:
        raise ValueError("No scenarios were found in the diagonal scan.")

    f_values = np.sort(diagonal["cut_f"].unique())
    x = np.arange(len(f_values), dtype=float) * 3.2
    x_left = x[0] - 1.35
    x_right = x[-1] + 1.35

    fig, (ax_impact, ax_catalog, ax_integrated) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11.5, 12.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0, 1.1]},
        constrained_layout=True,
    )

    all_impact_values = []
    all_integrated_values = []
    all_retention_values = []

    for scenario in scenarios:
        scenario_table = diagonal[
            diagonal["scenario"].astype(str) == scenario
        ].copy()

        color = scenario_colors.get(scenario, "0.25")
        scenario_label = scenario.capitalize()

        impact_values = metric_matrix(
            scenario_table,
            f_values,
            "impact_percent",
        )
        integrated_values = metric_matrix(
            scenario_table,
            f_values,
            "integrated_discarded_percent",
        )

        plot_metric_distribution(
            ax_impact,
            x,
            impact_values,
            color,
            f"{scenario_label} mean",
        )
        plot_metric_distribution(
            ax_integrated,
            x,
            integrated_values,
            color,
            f"{scenario_label} mean",
        )

        all_impact_values.append(impact_values.ravel())
        all_integrated_values.append(integrated_values.ravel())

        retention = aggregate_retention(
            scenario_table,
            f_values,
        )

        for population, linestyle, marker in (
            ("pointlike", "-", "o"),
            ("extended", "--", "s"),
        ):
            kept_percent = retention[
                f"{population}_kept_percent"
            ]

            ax_catalog.plot(
                x,
                kept_percent,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.1,
                label=(
                    f"{scenario_label} "
                    f"{population.capitalize()}"
                ),
                zorder=3,
            )
            all_retention_values.append(kept_percent)

    impact_positive = np.concatenate(all_impact_values)
    impact_positive = impact_positive[impact_positive > 0.0]

    if impact_positive.size == 0:
        raise ValueError("No positive map-impact values were found.")

    ax_impact.axhline(
        1.0,
        color="0.25",
        linewidth=1.4,
        linestyle="--",
        label=r"1\% reference",
    )
    ax_impact.axhline(
        10.0,
        color="0.45",
        linewidth=1.2,
        linestyle="-.",
        label=r"10\% reference",
    )
    ax_impact.set_yscale("log")
    ax_impact.set_ylim(
        0.5 * np.min(impact_positive),
        max(15.0, 1.8 * np.max(impact_positive)),
    )
    ax_impact.yaxis.set_major_formatter(
        FuncFormatter(percent_tick)
    )
    ax_impact.set_ylabel(
        "Conservative discarded-map peak\n"
        "/ final-map peak"
    )
    ax_impact.set_title(
        "Conservative map-level impact of discarded subhalos"
    )
    ax_impact.grid(True, which="both", alpha=0.25)
    ax_impact.legend(
        loc="lower right",
        ncol=2,
        fontsize=15.5,
        frameon=True,
    )

    integrated_positive = np.concatenate(
        all_integrated_values
    )
    integrated_positive = integrated_positive[
        integrated_positive > 0.0
    ]

    if integrated_positive.size == 0:
        raise ValueError(
            "No positive integrated-J values were found."
        )

    ax_integrated.set_yscale("log")
    ax_integrated.set_ylim(
        0.5 * np.min(integrated_positive),
        max(110.0, 1.25 * np.max(integrated_positive)),
    )
    ax_integrated.yaxis.set_major_formatter(
        FuncFormatter(percent_tick)
    )
    ax_integrated.set_ylabel(
        "Discarded integrated $J_s$\n"
        "/ full integrated $J_s$"
    )
    ax_integrated.set_title(
        "Catalogue-integrated J-factor removed by the cut"
    )
    ax_integrated.grid(True, which="both", alpha=0.25)
    ax_integrated.legend(
        loc="lower right",
        fontsize=15.5,
        frameon=True,
    )

    retention_positive = np.concatenate(
        all_retention_values
    )
    retention_positive = retention_positive[
        retention_positive > 0.0
    ]

    if retention_positive.size == 0:
        raise ValueError(
            "No positive catalog-retention values were found."
        )

    f_labels = [
        rf"$10^{{{int(np.rint(np.log10(value)))}}}$"
        for value in f_values
    ]

    ax_catalog.set_yscale("log")
    ax_catalog.set_ylim(
        0.3 * np.min(retention_positive),
        250.0,
    )
    ax_catalog.yaxis.set_major_formatter(
        FuncFormatter(percent_tick)
    )
    ax_integrated.set_xticks(x)
    ax_integrated.set_xticklabels(f_labels)
    ax_integrated.set_xlabel(
        r"$f$  ($J_{\rm cut}=f\,J_{\rm pixel,ref}$)"
    )
    ax_catalog.set_ylabel("Mean retained fraction")
    ax_catalog.set_title(
        "Catalogue retention by subhalo type"
    )
    ax_catalog.grid(True, which="both", alpha=0.25)
    ax_catalog.legend(
        loc="lower left",
        ncol=2,
        fontsize=15.5,
        frameon=True,
    )

    for axis in (ax_impact, ax_integrated, ax_catalog):
        axis.set_xlim(x_left, x_right)
        axis.set_xticks(x)
        axis.xaxis.set_minor_locator(NullLocator())

    fig.align_ylabels(
        (ax_impact, ax_catalog, ax_integrated)
    )

    nside_values = sorted(
        diagonal["nside"].astype(int).unique()
    )
    if len(nside_values) != 1:
        raise ValueError(
            "The combined CSV contains multiple NSIDE values."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "multi_repop_cut_diagnostic_"
        f"nside{nside_values[0]}_combined"
    )

    for output_format in formats:
        output_path = output_dir / f"{stem}.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}

        if output_format.lower() == "png":
            save_kwargs["dpi"] = dpi

        fig.savefig(output_path, **save_kwargs)
        print(f"Saved: {output_path}")

    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.combined_csv.is_file():
        raise FileNotFoundError(
            f"Combined CSV not found: {args.combined_csv}"
        )

    formats = [
        item.strip().lower()
        for item in args.formats.split(",")
        if item.strip()
    ]
    if not formats:
        raise ValueError("At least one output format is required.")

    table = pd.read_csv(args.combined_csv)
    validate_columns(table)
    diagonal = prepare_diagonal(table)

    plot_combined(
        diagonal=diagonal,
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
