#!/usr/bin/env python3
"""Plot correction diagnostics from the consolidated halo-level CSVs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "diagnostics" / "corrections" / "data"
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "diagnostics" / "corrections" / "plots" / "consolidated"
)
DEFAULT_THETA_BINS = 24
DEFAULT_HISTOGRAM_BINS = 60

SCENARIOS = ("fragile", "resilient")
STAGES = ("raw", "corrected")
SCENARIO_COLORS = {
    "fragile": "#EE3377",
    "resilient": "#009988",
}
SCENARIO_LINESTYLES = {
    "fragile": "-",
    "resilient": "--",
}

INPUT_COLUMNS = [
    "scenario",
    "repop_id",
    "halo_index",
    "halo_name",
    "theta_s_deg",
    "J_s",
    "J_raw",
    "R_raw",
    "rho_s_raw",
    "rho_s_corr",
    "rho_corr_over_raw",
    "J_corr",
    "R_corr",
    "npix_raw",
    "npix_corr",
]
PLOT_COLUMNS = [
    "scenario",
    "repop_id",
    "theta_s_deg",
    "R_raw",
    "R_corr",
]
SUMMARY_COLUMNS = [
    "scenario",
    "stage",
    "N",
    "median_R",
    "p16_R",
    "p84_R",
    "median_abs_R_minus_1",
    "fraction_abs_R_minus_1_lt_0p01",
    "fraction_abs_R_minus_1_lt_0p001",
]
BINNED_COLUMNS = [
    "scenario",
    "stage",
    "theta_lo_deg",
    "theta_hi_deg",
    "theta_center_deg",
    "n_halos",
    "n_repopulations",
    "median_log10_R",
    "p16_log10_R",
    "p84_log10_R",
]


class InputValidationError(RuntimeError):
    """A consolidated diagnostic CSV is absent or obviously malformed."""


@dataclass(frozen=True)
class ScenarioData:
    """Compact plotting columns for one correction scenario."""

    scenario: str
    repop_id: np.ndarray
    theta_s_deg: np.ndarray
    r_raw: np.ndarray
    r_corr: np.ndarray

    def ratios(self, stage: str) -> np.ndarray:
        if stage == "raw":
            return self.r_raw
        if stage == "corrected":
            return self.r_corr
        raise ValueError(f"Unknown correction stage: {stage!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw/corrected CLUMPY J-factor mismatch diagnostics from "
            "the two consolidated scenario CSVs."
        )
    )
    parser.add_argument(
        "--fragile-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "corrections_fragile_repop0000_0021.csv",
    )
    parser.add_argument(
        "--resilient-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "corrections_resilient_repop0000_0021.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--theta-bins", type=int, default=DEFAULT_THETA_BINS)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.theta_bins <= 0:
        raise ValueError("--theta-bins must be a positive integer.")
    if len(args.formats) != len(set(args.formats)):
        raise ValueError("--formats must not contain duplicates.")


def _require_nonempty_file(path: Path) -> None:
    if not path.is_file():
        raise InputValidationError(f"Input CSV is missing: {path}")
    if path.stat().st_size == 0:
        raise InputValidationError(f"Input CSV is empty: {path}")


def _require_finite_positive(values: np.ndarray, label: str, path: Path) -> None:
    invalid = (~np.isfinite(values)) | (values <= 0.0)
    if np.any(invalid):
        raise InputValidationError(
            f"{path}: {label} contains {np.count_nonzero(invalid)} "
            "non-finite/non-positive value(s)."
        )


def load_scenario_csv(path: Path, expected_scenario: str) -> ScenarioData:
    """Validate a consolidated CSV and load only the plotting columns."""
    path = Path(path)
    if expected_scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {expected_scenario!r}")
    _require_nonempty_file(path)

    try:
        header = pd.read_csv(path, nrows=0)
    except (OSError, pd.errors.ParserError) as error:
        raise InputValidationError(
            f"Could not read CSV header {path}: {error}"
        ) from error
    actual_columns = list(header.columns)
    if actual_columns != INPUT_COLUMNS:
        raise InputValidationError(
            f"{path}: schema mismatch; expected {INPUT_COLUMNS}, "
            f"found {actual_columns}."
        )

    try:
        table = pd.read_csv(
            path,
            usecols=PLOT_COLUMNS,
            dtype={
                "scenario": "category",
                "repop_id": "float64",
                "theta_s_deg": "float64",
                "R_raw": "float64",
                "R_corr": "float64",
            },
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise InputValidationError(
            f"Could not read plotting data {path}: {error}"
        ) from error
    if table.empty:
        raise InputValidationError(f"Input CSV has no halo rows: {path}")

    actual_scenarios = set(table["scenario"].dropna().tolist())
    if actual_scenarios != {expected_scenario} or table["scenario"].isna().any():
        raise InputValidationError(
            f"{path}: expected only scenario {expected_scenario!r}, "
            f"found {sorted(actual_scenarios)}."
        )

    repop_id = table["repop_id"].to_numpy(dtype=np.float64, copy=False)
    invalid_repop = (
        (~np.isfinite(repop_id))
        | (repop_id < 0.0)
        | (repop_id != np.floor(repop_id))
        | (repop_id > np.iinfo(np.int32).max)
    )
    if np.any(invalid_repop):
        raise InputValidationError(
            f"{path}: repop_id contains {np.count_nonzero(invalid_repop)} "
            "invalid value(s)."
        )

    theta_s_deg = table["theta_s_deg"].to_numpy(dtype=np.float64, copy=True)
    r_raw = table["R_raw"].to_numpy(dtype=np.float64, copy=True)
    r_corr = table["R_corr"].to_numpy(dtype=np.float64, copy=True)
    _require_finite_positive(theta_s_deg, "theta_s_deg", path)
    _require_finite_positive(r_raw, "R_raw", path)
    _require_finite_positive(r_corr, "R_corr", path)

    return ScenarioData(
        scenario=expected_scenario,
        repop_id=repop_id.astype(np.int32, copy=True),
        theta_s_deg=theta_s_deg,
        r_raw=r_raw,
        r_corr=r_corr,
    )


def log10_ratios(data: ScenarioData, stage: str) -> np.ndarray:
    """Return log10(J_rendered/J_s) for one validated stage."""
    return np.log10(data.ratios(stage))


def common_histogram_edges(
    arrays: Sequence[np.ndarray], n_bins: int = DEFAULT_HISTOGRAM_BINS
) -> np.ndarray:
    """Construct common full-range linear edges in log-ratio space."""
    if n_bins <= 0:
        raise ValueError("Histogram bin count must be positive.")
    if not arrays or any(np.asarray(values).size == 0 for values in arrays):
        raise ValueError("Histogram inputs must be non-empty.")
    lower = min(0.0, *(float(np.min(values)) for values in arrays))
    upper = max(0.0, *(float(np.max(values)) for values in arrays))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Histogram inputs must be finite.")
    if lower == upper:
        half_width = max(0.05, 0.05 * abs(lower))
        lower -= half_width
        upper += half_width
    return np.linspace(lower, upper, n_bins + 1)


def normalized_histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the fraction of scenario halos in each bin."""
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(np.float64) / values.size


def common_theta_edges(
    data_by_scenario: dict[str, ScenarioData], n_bins: int
) -> np.ndarray:
    """Construct common full-range logarithmic theta_s edges."""
    if n_bins <= 0:
        raise ValueError("Theta bin count must be positive.")
    lower = min(float(np.min(data.theta_s_deg)) for data in data_by_scenario.values())
    upper = max(float(np.max(data.theta_s_deg)) for data in data_by_scenario.values())
    if lower <= 0.0 or not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("theta_s_deg must be finite and positive.")
    if lower == upper:
        lower /= np.sqrt(10.0)
        upper *= np.sqrt(10.0)
    return np.geomspace(lower, upper, n_bins + 1)


def build_distribution_summary(
    data_by_scenario: dict[str, ScenarioData],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        data = data_by_scenario[scenario]
        for stage in STAGES:
            ratios = data.ratios(stage)
            p16, median, p84 = np.percentile(ratios, [16.0, 50.0, 84.0])
            absolute_residual = np.abs(ratios - 1.0)
            rows.append(
                {
                    "scenario": scenario,
                    "stage": stage,
                    "N": int(ratios.size),
                    "median_R": float(median),
                    "p16_R": float(p16),
                    "p84_R": float(p84),
                    "median_abs_R_minus_1": float(np.median(absolute_residual)),
                    "fraction_abs_R_minus_1_lt_0p01": float(
                        np.mean(absolute_residual < 0.01)
                    ),
                    "fraction_abs_R_minus_1_lt_0p001": float(
                        np.mean(absolute_residual < 0.001)
                    ),
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _theta_bin_indices(theta_s_deg: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, theta_s_deg, side="right") - 1
    indices[theta_s_deg == edges[-1]] = len(edges) - 2
    return indices


def build_binned_theta_summary(
    data_by_scenario: dict[str, ScenarioData], edges: np.ndarray
) -> pd.DataFrame:
    """Calculate pooled halo percentiles on one common theta_s grid."""
    centers = np.sqrt(edges[:-1] * edges[1:])
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        data = data_by_scenario[scenario]
        bin_indices = _theta_bin_indices(data.theta_s_deg, edges)
        stage_values = {stage: log10_ratios(data, stage) for stage in STAGES}
        for bin_index, center in enumerate(centers):
            selected = bin_indices == bin_index
            n_halos = int(np.count_nonzero(selected))
            n_repopulations = (
                int(np.unique(data.repop_id[selected]).size) if n_halos else 0
            )
            for stage in STAGES:
                if n_halos:
                    p16, median, p84 = np.percentile(
                        stage_values[stage][selected], [16.0, 50.0, 84.0]
                    )
                else:
                    p16 = median = p84 = np.nan
                rows.append(
                    {
                        "scenario": scenario,
                        "stage": stage,
                        "theta_lo_deg": float(edges[bin_index]),
                        "theta_hi_deg": float(edges[bin_index + 1]),
                        "theta_center_deg": float(center),
                        "n_halos": n_halos,
                        "n_repopulations": n_repopulations,
                        "median_log10_R": float(median),
                        "p16_log10_R": float(p16),
                        "p84_log10_R": float(p84),
                    }
                )
    result = pd.DataFrame(rows, columns=BINNED_COLUMNS)
    return result.sort_values(
        ["scenario", "stage", "theta_lo_deg"], ignore_index=True
    )


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
        }
    )


def _finish_axis(axis: plt.Axes) -> None:
    axis.grid(False, which="both")
    axis.tick_params(direction="in", which="both", top=True, right=True)
    axis.legend(frameon=False)


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for output_format in formats:
        path = output_dir / f"{stem}.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = 300
        figure.savefig(path, **save_kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def make_distribution_figure(
    data_by_scenario: dict[str, ScenarioData],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    configure_plot_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.6), constrained_layout=True
    )
    for axis, stage in zip(axes, STAGES):
        values_by_scenario = {
            scenario: log10_ratios(data_by_scenario[scenario], stage)
            for scenario in SCENARIOS
        }
        edges = common_histogram_edges(list(values_by_scenario.values()))
        for scenario in SCENARIOS:
            fractions = normalized_histogram(values_by_scenario[scenario], edges)
            axis.stairs(
                fractions,
                edges,
                color=SCENARIO_COLORS[scenario],
                linestyle=SCENARIO_LINESTYLES[scenario],
                linewidth=1.9,
                label=scenario.capitalize(),
            )
        axis.axvline(0.0, color="0.4", linestyle=":", linewidth=1.2)
        axis.set_xlim(edges[0], edges[-1])
        axis.set_xlabel(
            r"$\log_{10}(J_{\rm raw}/J_s)$"
            if stage == "raw"
            else r"$\log_{10}(J_{\rm corr}/J_s)$"
        )
        axis.set_ylabel("Fraction of halos")
        axis.set_title("Raw" if stage == "raw" else "Corrected")
        _finish_axis(axis)
    return _save_figure(
        figure,
        output_dir,
        "correction_log_ratio_distributions",
        formats,
    )


def make_theta_figure(
    binned: pd.DataFrame,
    edges: np.ndarray,
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    configure_plot_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.8), constrained_layout=True, sharex=True
    )
    for axis, stage in zip(axes, STAGES):
        for scenario in SCENARIOS:
            selected = binned.loc[
                (binned["scenario"] == scenario) & (binned["stage"] == stage)
            ]
            x = selected["theta_center_deg"].to_numpy(dtype=np.float64)
            median = selected["median_log10_R"].to_numpy(dtype=np.float64)
            lower = selected["p16_log10_R"].to_numpy(dtype=np.float64)
            upper = selected["p84_log10_R"].to_numpy(dtype=np.float64)
            color = SCENARIO_COLORS[scenario]
            axis.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.16,
                linewidth=0.0,
            )
            axis.plot(
                x,
                median,
                color=color,
                linestyle=SCENARIO_LINESTYLES[scenario],
                linewidth=1.9,
                label=scenario.capitalize(),
            )
        axis.axhline(0.0, color="0.4", linestyle=":", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_xlim(edges[0], edges[-1])
        axis.set_xlabel(r"$\theta_s\;[\mathrm{deg}]$")
        axis.set_ylabel(
            r"$\log_{10}(J_{\rm raw}/J_s)$"
            if stage == "raw"
            else r"$\log_{10}(J_{\rm corr}/J_s)$"
        )
        axis.set_title("Raw" if stage == "raw" else "Corrected")
        _finish_axis(axis)
    figure.text(
        0.5,
        -0.015,
        "Shaded bands show the 16th–84th percentile halo-to-halo dispersion.",
        ha="center",
        fontsize=11,
    )
    return _save_figure(
        figure,
        output_dir,
        "correction_log_ratio_vs_theta",
        formats,
    )


def write_derived_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def print_distribution_summary(summary: pd.DataFrame) -> None:
    print("Distribution summary")
    for row in summary.itertuples(index=False):
        print(
            f"{row.scenario:9s} {row.stage:9s} N={row.N} "
            f"median_R={row.median_R:.8g} "
            f"p16_R={row.p16_R:.8g} p84_R={row.p84_R:.8g} "
            f"median_abs(R-1)={row.median_abs_R_minus_1:.8g} "
            f"frac_abs(R-1)<0.01="
            f"{row.fraction_abs_R_minus_1_lt_0p01:.8g} "
            f"frac_abs(R-1)<0.001="
            f"{row.fraction_abs_R_minus_1_lt_0p001:.8g}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        validate_cli_args(args)
        data_by_scenario = {
            "fragile": load_scenario_csv(args.fragile_csv, "fragile"),
            "resilient": load_scenario_csv(args.resilient_csv, "resilient"),
        }
        summary = build_distribution_summary(data_by_scenario)
        theta_edges = common_theta_edges(data_by_scenario, args.theta_bins)
        binned = build_binned_theta_summary(data_by_scenario, theta_edges)

        data_output_dir = args.output_dir / "data"
        summary_path = data_output_dir / "correction_distribution_summary.csv"
        binned_path = data_output_dir / "correction_ratio_vs_theta_binned.csv"
        write_derived_csv(summary, summary_path)
        write_derived_csv(binned, binned_path)
        figure_paths = make_distribution_figure(
            data_by_scenario, args.output_dir, args.formats
        )
        figure_paths.extend(
            make_theta_figure(binned, theta_edges, args.output_dir, args.formats)
        )
    except (InputValidationError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print_distribution_summary(summary)
    print(f"Saved: {summary_path}")
    print(f"Saved: {binned_path}")
    for path in figure_paths:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
