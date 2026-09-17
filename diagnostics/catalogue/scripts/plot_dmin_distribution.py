#!/usr/bin/env python3
"""Plot the minimum DGC distance across 500 repopulations per scenario."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
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


SCENARIOS = ("fragile", "resilient")
SCENARIO_COLORS = {
    "fragile": "#EE3377",
    "resilient": "#009988",
}
EXPECTED_REPOPULATIONS = 500
REQUIRED_COLUMNS = {"scenario", "repop_id", "min_dgc_kpc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot minimum DGC-distance distributions for the fragile and "
            "resilient 500-repopulation catalogues."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fragile-n-bins",
        type=int,
        default=30,
        help="Number of narrow linear bins for fragile (default: 30).",
    )
    parser.add_argument(
        "--resilient-n-bins",
        type=int,
        default=30,
        help="Number of logarithmic bins for resilient (default: 30).",
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
        item.strip().lower() for item in text.split(",") if item.strip()
    )
    invalid = sorted(set(formats) - {"png", "pdf"})
    if not formats:
        raise ValueError("At least one output format is required.")
    if invalid:
        raise ValueError("Unsupported format(s): " + ", ".join(invalid))
    return formats


def load_dmin_catalogue(path: Path) -> dict[str, np.ndarray]:
    """Read and validate D_min values grouped by disruption scenario."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    values: dict[str, list[float]] = {scenario: [] for scenario in SCENARIOS}
    repop_ids: dict[str, set[int]] = {scenario: set() for scenario in SCENARIOS}

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Input CSV is missing columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            scenario = row["scenario"]
            if scenario not in SCENARIOS:
                raise ValueError(
                    f"Unsupported scenario on line {line_number}: {scenario!r}"
                )
            try:
                repop_id = int(row["repop_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid repop_id on line {line_number}: "
                    f"{row['repop_id']!r}"
                ) from error
            if repop_id in repop_ids[scenario]:
                raise ValueError(
                    f"Duplicate repop_id for {scenario}: {repop_id}"
                )
            try:
                dmin = float(row["min_dgc_kpc"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid min_dgc_kpc on line {line_number}: "
                    f"{row['min_dgc_kpc']!r}"
                ) from error
            if not np.isfinite(dmin) or dmin <= 0.0:
                raise ValueError(
                    "min_dgc_kpc must be finite and positive on line "
                    f"{line_number}."
                )

            repop_ids[scenario].add(repop_id)
            values[scenario].append(dmin)

    for scenario in SCENARIOS:
        actual = len(repop_ids[scenario])
        if actual != EXPECTED_REPOPULATIONS:
            raise ValueError(
                f"Scenario {scenario!r} must contain "
                f"{EXPECTED_REPOPULATIONS} unique repopulations; found {actual}."
            )

    return {
        scenario: np.asarray(values[scenario], dtype=np.float64)
        for scenario in SCENARIOS
    }


def _validate_n_bins(n_bins: int) -> None:
    if (
        isinstance(n_bins, (bool, np.bool_))
        or not isinstance(n_bins, (int, np.integer))
        or n_bins <= 0
    ):
        raise ValueError("Number of bins must be a positive integer.")


def make_fragile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return narrow linear bin edges for the tightly clustered fragile data."""
    _validate_n_bins(n_bins)
    values = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("min_dgc_kpc values must be finite and positive.")

    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = 0.05 * (upper - lower)
    if padding == 0.0:
        padding = max(abs(lower), 1.0) * 1.0e-6
    return np.linspace(lower - padding, upper + padding, n_bins + 1)


def make_resilient_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return logarithmic bin edges for the broad resilient distribution."""
    _validate_n_bins(n_bins)
    values = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("min_dgc_kpc values must be finite and positive.")

    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        lower /= np.sqrt(10.0)
        upper *= np.sqrt(10.0)
    return np.geomspace(lower, upper, n_bins + 1)


def _format_axis(axis, scenario: str) -> None:
    axis.set_title(f"{scenario.capitalize()} scenario")
    axis.set_xlabel(r"$D_{\rm min}\,[\mathrm{kpc}]$")
    axis.set_ylabel("Number of repopulations")
    axis.set_ylim(bottom=0)
    axis.tick_params(axis="both", which="both", labelsize=15)
    axis.xaxis.get_offset_text().set_fontsize(15)
    axis.yaxis.get_offset_text().set_fontsize(15)
    axis.grid(False, which="both")


def plot_dmin_distribution(
    values: dict[str, np.ndarray],
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    fragile_n_bins: int,
    resilient_n_bins: int,
) -> list[Path]:
    """Draw and save one panel per scenario using scenario-specific bins."""
    bins = {
        "fragile": make_fragile_bins(values["fragile"], fragile_n_bins),
        "resilient": make_resilient_bins(
            values["resilient"], resilient_n_bins
        ),
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 5.2),
        constrained_layout=True,
    )
    for axis, scenario in zip(axes, SCENARIOS):
        axis.hist(
            values[scenario],
            bins=bins[scenario],
            density=False,
            histtype="stepfilled",
            color=SCENARIO_COLORS[scenario],
            alpha=0.78,
        )
        if scenario == "resilient":
            axis.set_xscale("log")
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        _format_axis(axis, scenario)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for output_format in formats:
        output_path = output_dir / f"dmin_distribution.{output_format}"
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
    values = load_dmin_catalogue(args.input_csv)
    plot_dmin_distribution(
        values=values,
        output_dir=args.output_dir,
        formats=parse_formats(args.formats),
        dpi=args.dpi,
        fragile_n_bins=args.fragile_n_bins,
        resilient_n_bins=args.resilient_n_bins,
    )


if __name__ == "__main__":
    main()
