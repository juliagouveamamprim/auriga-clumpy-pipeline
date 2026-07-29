#!/usr/bin/env python3
"""Plot multi-repopulation catalog-cut diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullLocator


REQUIRED_COLUMNS = {
    "scenario",
    "repop_id",
    "nside",
    "pointlike_f",
    "extended_f",
    "ratio_max_discarded_theta_s_envelope_to_final",
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
    return f"{value:g}%"


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


def plot_scenario(
    diagonal: pd.DataFrame,
    scenario: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    scenario_table = diagonal[
        diagonal["scenario"].astype(str) == scenario
    ].copy()

    if scenario_table.empty:
        raise ValueError(f"No rows found for scenario {scenario!r}.")

    f_values = np.sort(scenario_table["cut_f"].unique())
    x = np.arange(len(f_values), dtype=float) * 3.2
    x_left = x[0] - 1.35
    x_right = x[-1] + 1.35

    impact = scenario_table.pivot(
        index="repop_id",
        columns="cut_f",
        values="impact_percent",
    ).reindex(columns=f_values)

    if impact.isna().any().any():
        raise ValueError(
            f"Incomplete diagonal scan found for scenario {scenario!r}."
        )

    impact_values = impact.to_numpy(dtype=float)
    impact_mean = np.mean(impact_values, axis=0)
    impact_p16 = np.percentile(impact_values, 16.0, axis=0)
    impact_p84 = np.percentile(impact_values, 84.0, axis=0)
    retention = aggregate_retention(scenario_table, f_values)

    mean_color = "#1F77B4"

    fig, (ax_impact, ax_catalog) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11.5, 9.3),
        gridspec_kw={"height_ratios": [1.15, 1.0]},
        constrained_layout=True,
    )

    # Conservative combined impact.
    for repop_values in impact_values:
        ax_impact.plot(
            x,
            repop_values,
            color="0.52",
            linewidth=1.0,
            alpha=0.70,
            zorder=1,
        )

    ax_impact.fill_between(
        x,
        impact_p16,
        impact_p84,
        color=mean_color,
        alpha=0.20,
        label="16–84% range",
        zorder=2,
    )
    ax_impact.plot(
        x,
        impact_mean,
        marker="o",
        linewidth=2.1,
        color=mean_color,
        label="Mean over repopulations",
        zorder=3,
    )
    ax_impact.axhline(
        1.0,
        color="0.25",
        linewidth=1.4,
        linestyle="--",
        label="1% threshold",
    )
    ax_impact.axhline(
        10.0,
        color="0.45",
        linewidth=1.2,
        linestyle="-.",
        label="10% reference level",
    )

    positive_min = np.min(impact_values[impact_values > 0.0])
    upper_limit = max(15.0, 1.8 * np.max(impact_values))

    ax_impact.set_yscale("log")
    ax_impact.set_ylim(0.5 * positive_min, upper_limit)
    ax_impact.yaxis.set_major_formatter(FuncFormatter(percent_tick))
    ax_impact.set_ylabel(
        "Conservative discarded-map peak\n"
        "/ final-map peak"
    )
    ax_impact.set_title(
        f"{scenario.capitalize()}: impact of discarded subhalos"
    )
    ax_impact.grid(True, which="both", alpha=0.25)
    ax_impact.legend(
        loc="lower right",
        ncol=2,
        fontsize=9,
        frameon=True,
    )
    ax_impact.set_xlim(x_left, x_right)
    ax_impact.set_xticks(x)
    ax_impact.set_xticklabels([])
    ax_impact.xaxis.set_minor_locator(NullLocator())

    # Catalog retention.
    group_x = x
    pair_offset = 0.18

    pointlike_x = group_x - pair_offset
    extended_x = group_x + pair_offset

    pointlike_kept = retention["pointlike_kept_percent"]
    extended_kept = retention["extended_kept_percent"]

    pointlike_color = "#111111"
    extended_color = "#D62728"

    ax_catalog.plot(
        pointlike_x,
        pointlike_kept,
        marker="o",
        linewidth=2.0,
        color=pointlike_color,
        label="Pointlike",
        zorder=3,
    )
    ax_catalog.plot(
        extended_x,
        extended_kept,
        marker="s",
        linewidth=2.0,
        color=extended_color,
        label="Extended",
        zorder=3,
    )

    for population, positions, kept_percent, color, x_offset, alignment in (
        (
            "pointlike",
            pointlike_x,
            pointlike_kept,
            pointlike_color,
            -9,
            "right",
        ),
        (
            "extended",
            extended_x,
            extended_kept,
            extended_color,
            9,
            "left",
        ),
    ):
        kept_count = retention[f"{population}_kept_count"]
        discarded_count = retention[f"{population}_discarded_count"]

        for index, (
            position,
            kept_pct,
            kept_n,
            discarded_n,
        ) in enumerate(
            zip(
                positions,
                kept_percent,
                kept_count,
                discarded_count,
                strict=True,
            )
        ):
            # Keep labels below, except the final pointlike label.
            if population == "pointlike" and index == len(positions) - 1:
                label_x_offset = 9
                label_y_offset = 9
                label_alignment = "left"
                vertical_alignment = "bottom"
            else:
                label_x_offset = x_offset
                label_y_offset = -10
                label_alignment = alignment
                vertical_alignment = "top"

            ax_catalog.annotate(
                (
                    f"K {compact_count(kept_n)}\n"
                    f"D {compact_count(discarded_n)}"
                ),
                xy=(position, kept_pct),
                xytext=(label_x_offset, label_y_offset),
                textcoords="offset points",
                ha=label_alignment,
                va=vertical_alignment,
                fontsize=7.5,
                color=color,
                linespacing=1.15,
                annotation_clip=False,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.5,
                },
            )

    f_labels = [
        rf"$10^{{{int(np.rint(np.log10(value)))}}}$"
        for value in f_values
    ]

    positive_retention = np.concatenate(
        [pointlike_kept, extended_kept]
    )
    positive_retention = positive_retention[
        positive_retention > 0.0
    ]

    ax_catalog.set_yscale("log")
    ax_catalog.set_ylim(
        0.3 * np.min(positive_retention),
        250.0,
    )
    ax_catalog.yaxis.set_major_formatter(
        FuncFormatter(percent_tick)
    )

    ax_catalog.set_xticks(group_x)
    ax_catalog.set_xticklabels(f_labels)
    ax_catalog.set_xlim(x_left, x_right)
    ax_catalog.set_xlabel(
        r"$f$  ($J_{\rm cut}=f\,J_{\rm pixel,ref}$)"
    )
    ax_catalog.set_ylabel("Mean retained fraction")
    ax_catalog.set_title(
        "Catalog retention by subhalo type"
    )
    ax_catalog.grid(True, which="both", alpha=0.25)
    ax_catalog.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
    )

    fig.align_ylabels((ax_impact, ax_catalog))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"multi_repop_cut_diagnostic_nside2048_{scenario}"

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

    scenarios = sorted(diagonal["scenario"].astype(str).unique())
    for scenario in scenarios:
        plot_scenario(
            diagonal=diagonal,
            scenario=scenario,
            output_dir=args.output_dir,
            formats=formats,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
