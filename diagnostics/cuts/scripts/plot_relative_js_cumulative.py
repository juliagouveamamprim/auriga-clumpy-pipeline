#!/usr/bin/env python3

"""Plot cumulative relative-Js distributions from a saved scan."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "diagnostics" / "cuts" / "plots"
SCENARIO_ORDER = ("fragile", "resilient")
QUANTILES_TO_REPORT = (0.50, 0.90, 0.99)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot saved cumulative distributions of Js/Js,max."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        required=True,
        help="NPZ produced by scan_relative_js_cumulative.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Output basename without extension (default: input basename).",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats (default: png,pdf).",
    )
    return parser.parse_args()


def parse_formats(text):
    formats = tuple(
        item.strip().lower()
        for item in text.split(",")
        if item.strip()
    )
    valid = {"png", "pdf"}
    invalid = sorted(set(formats) - valid)

    if not formats:
        raise ValueError("At least one output format is required.")
    if invalid:
        raise ValueError("Unsupported format(s): " + ", ".join(invalid))

    return formats


def percentile_band(array):
    return (
        np.percentile(array, 16, axis=0),
        np.percentile(array, 50, axis=0),
        np.percentile(array, 84, axis=0),
    )


def x_at_cumulative_fraction(x_values, cdf, target):
    if target <= cdf[0]:
        return x_values[0]
    if target >= cdf[-1]:
        return x_values[-1]

    index = np.searchsorted(cdf, target, side="left")
    x0 = np.log10(x_values[index - 1])
    x1 = np.log10(x_values[index])
    y0 = cdf[index - 1]
    y1 = cdf[index]

    if y1 == y0:
        return x_values[index]

    fraction = (target - y0) / (y1 - y0)
    return 10.0 ** (x0 + fraction * (x1 - x0))


def scenarios_in_file(data):
    if "scenarios" in data.files:
        return tuple(str(item) for item in data["scenarios"])

    # Compatibility with the cache produced by the original Lattes script.
    return tuple(
        scenario
        for scenario in SCENARIO_ORDER
        if f"{scenario}_cdfs" in data.files
    )


def default_basename(input_path):
    basename = input_path.stem
    if basename.endswith("_cache"):
        basename = basename[: -len("_cache")]
    return basename


def main():
    args = parse_args()
    formats = parse_formats(args.formats)

    if not args.input_npz.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {args.input_npz}")

    with np.load(args.input_npz, allow_pickle=False) as data:
        x_values = np.asarray(data["xvals"], dtype=np.float64)
        scenarios = scenarios_in_file(data)
        cdfs = {
            scenario: np.asarray(
                data[f"{scenario}_cdfs"],
                dtype=np.float64,
            )
            for scenario in scenarios
        }

    if not scenarios:
        raise ValueError("No scenario CDF arrays were found in the input NPZ.")

    for scenario, values in cdfs.items():
        if values.ndim != 2 or values.shape[1] != len(x_values):
            raise ValueError(
                f"Invalid {scenario} CDF shape {values.shape}; "
                f"expected (n_catalogues, {len(x_values)})."
            )

    colors = {
        "fragile": "#0072B2",
        "resilient": "#D55E00",
    }
    figure, axis = plt.subplots(figsize=(6.4, 4.3))

    for scenario in scenarios:
        lower, median, upper = percentile_band(cdfs[scenario])
        color = colors.get(scenario, None)
        label = scenario.capitalize()

        axis.plot(
            x_values,
            median,
            color=color,
            linewidth=1.8,
            label=f"{label} median",
        )
        axis.fill_between(
            x_values,
            lower,
            upper,
            color=color,
            alpha=0.18,
            label=f"{label} 16-84 percentile",
        )

        for quantile in QUANTILES_TO_REPORT:
            x_quantile = x_at_cumulative_fraction(
                x_values,
                median,
                quantile,
            )
            print(
                f"{scenario:9s} median CDF reaches "
                f"{100.0 * quantile:5.1f}% at "
                f"Js/Js,max = {x_quantile:.3e}"
            )

            if np.isclose(quantile, 0.99):
                axis.axvline(
                    x_quantile,
                    color=color,
                    linestyle=":",
                    linewidth=1.3,
                    alpha=0.9,
                )
    axis.set_xscale("log")
    axis.set_xlim(1.0e-12, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel(r"$J_s/J_{s,\max}^{\rm cat}$")
    axis.set_ylabel("Cumulative fraction of subhalos")
    axis.grid(alpha=0.25, which="major")

    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="0.45",
            linestyle=":",
            linewidth=1.3,
        )
    )
    labels.append(r"99\% of subhalos (median CDF)")
    axis.legend(
        handles,
        labels,
        frameon=False,
        loc="lower right",
        fontsize=8,
    )

    figure.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.basename or default_basename(args.input_npz)

    for output_format in formats:
        output_path = args.output_dir / f"{basename}.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = 200
        figure.savefig(output_path, **save_kwargs)
        print(f"Saved: {output_path}")

    plt.close(figure)


if __name__ == "__main__":
    main()
