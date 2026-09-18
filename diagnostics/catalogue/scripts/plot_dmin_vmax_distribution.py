#!/usr/bin/env python3
"""Plot Vmax for the subhalo defining D_min in each repopulation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_REPOPULATIONS = 500
REPOP_IDS = range(EXPECTED_REPOPULATIONS)
VALID_SCENARIOS = ("fragile", "resilient")
SCENARIO_COLORS = {
    "fragile": "#EE3377",
    "resilient": "#009988",
}
REQUIRED_CSV_COLUMNS = {"scenario", "repop_id", "min_dgc_kpc"}
REQUIRED_DATA_COLUMNS = {"Vmax", "Xearth", "Yearth", "Zearth"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Vmax for the unique subhalo at minimum galactocentric "
            "distance in each 500-repopulation catalogue."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs",
        help=(
            "Directory containing repop_XXXX/fullrepop_hydro_<scenario>.h5 "
            f"files (default: {REPOSITORY_ROOT / 'outputs'})."
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs"
        / "diagnostics"
        / "full_catalogue_dgc_0500.csv",
        help="CSV containing scenario, repop_id, and min_dgc_kpc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "diagnostics"
        / "catalogue"
        / "plots"
        / "dmin_vmax_distribution",
        help="Directory for dmin_vmax_distribution.png and .pdf.",
    )
    parser.add_argument(
        "--scenario",
        choices=VALID_SCENARIOS,
        default="resilient",
        help="Disruption scenario to process (default: resilient).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=30,
        help="Number of linear Vmax histogram bins (default: 30).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of HDF5 rows read per chunk (default: 1000000).",
    )
    parser.add_argument(
        "--dmin-rtol",
        type=float,
        default=1.0e-10,
        help="Relative tolerance for the HDF5/CSV D_min check.",
    )
    parser.add_argument(
        "--dmin-atol-kpc",
        type=float,
        default=1.0e-10,
        help="Absolute tolerance in kpc for the HDF5/CSV D_min check.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if not np.isfinite(args.dmin_rtol) or args.dmin_rtol < 0.0:
        raise ValueError("--dmin-rtol must be finite and non-negative.")
    if not np.isfinite(args.dmin_atol_kpc) or args.dmin_atol_kpc < 0.0:
        raise ValueError("--dmin-atol-kpc must be finite and non-negative.")


def catalogue_path(input_root: Path, repop_id: int, scenario: str) -> Path:
    return (
        input_root
        / f"repop_{repop_id:04d}"
        / f"fullrepop_hydro_{scenario}.h5"
    )


def validate_catalogue_files(input_root: Path, scenario: str) -> None:
    """Require one HDF5 catalogue for every expected repopulation."""
    paths = [
        catalogue_path(input_root, repop_id, scenario) for repop_id in REPOP_IDS
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        remaining = len(missing) - min(len(missing), 10)
        suffix = "" if remaining == 0 else f"\n  ... and {remaining} more"
        raise FileNotFoundError(
            f"Scenario {scenario!r} is missing {len(missing)} of "
            f"{EXPECTED_REPOPULATIONS} HDF5 catalogues:\n{preview}{suffix}"
        )


def load_dmin_values(path: Path, scenario: str) -> dict[int, float]:
    """Read one D_min per repopulation for the requested scenario."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    dmin_by_repop: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_CSV_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Input CSV is missing columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            if row["scenario"] != scenario:
                continue
            try:
                repop_id = int(row["repop_id"])
                dmin = float(row["min_dgc_kpc"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {scenario} row on line {line_number}: {row}"
                ) from error
            if repop_id in dmin_by_repop:
                raise ValueError(
                    f"Duplicate {scenario} repop_id in CSV: {repop_id}"
                )
            if not np.isfinite(dmin) or dmin <= 0.0:
                raise ValueError(
                    f"min_dgc_kpc must be finite and positive on line "
                    f"{line_number}."
                )
            dmin_by_repop[repop_id] = dmin

    expected_ids = set(REPOP_IDS)
    actual_ids = set(dmin_by_repop)
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)
        unexpected_ids = sorted(actual_ids - expected_ids)
        details = []
        if missing_ids:
            details.append(f"missing IDs: {missing_ids[:10]}")
        if unexpected_ids:
            details.append(f"unexpected IDs: {unexpected_ids[:10]}")
        raise ValueError(
            f"Scenario {scenario!r} must contain exactly "
            f"{EXPECTED_REPOPULATIONS} repopulations (IDs 0--499); "
            + "; ".join(details)
        )
    return dmin_by_repop


def decode_column_names(raw_names: np.ndarray) -> list[str]:
    return [
        name.decode() if isinstance(name, bytes) else str(name)
        for name in raw_names
    ]


def read_earth_position(handle: h5py.File, path: Path) -> np.ndarray:
    """Read the observer position saved alongside this catalogue."""
    location = "inputs/host/position_Earth/value"
    if location not in handle:
        raise KeyError(f"Expected dataset {location!r} not found in {path}")
    position = np.asarray(handle[location][...], dtype=np.float64)
    if position.shape != (3,) or np.any(~np.isfinite(position)):
        raise ValueError(
            f"Invalid observer position in {path}: expected three finite kpc values."
        )
    return position


def galactocentric_distance_kpc(
    xearth: np.ndarray,
    yearth: np.ndarray,
    zearth: np.ndarray,
    earth_position_kpc: np.ndarray,
) -> np.ndarray:
    """Recover D_GC from the Earth-centred convention written by the pipeline.

    The HDF5 writer stores Xearth = -(Xgc - Xearth_position),
    Yearth = Ygc - Yearth_position, and Zearth = Zgc - Zearth_position.
    """
    xgc = earth_position_kpc[0] - xearth
    ygc = earth_position_kpc[1] + yearth
    zgc = earth_position_kpc[2] + zearth
    return np.sqrt(xgc**2 + ygc**2 + zgc**2)


def find_dmin_subhalo_vmax(path: Path, chunk_size: int) -> tuple[float, float]:
    """Return (D_min, Vmax) after scanning one HDF5 catalogue in chunks."""
    best_dgc = np.inf
    best_vmax = np.nan
    n_at_best = 0

    with h5py.File(path, "r") as handle:
        dataset_name = "iteration_0/data"
        if dataset_name not in handle:
            raise KeyError(f"Expected dataset {dataset_name!r} not found in {path}")
        dataset = handle[dataset_name]
        if dataset.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional dataset in {path}; found "
                f"shape {dataset.shape}."
            )
        if "column_names" not in dataset.attrs:
            raise KeyError(f"Missing column_names attribute in {path}")
        column_names = decode_column_names(dataset.attrs["column_names"])
        missing = sorted(REQUIRED_DATA_COLUMNS - set(column_names))
        if missing:
            raise ValueError(f"Dataset in {path} is missing columns: {missing}")
        if dataset.shape[1] != len(column_names):
            raise ValueError(
                f"Column metadata does not match dataset shape in {path}: "
                f"{len(column_names)} names for {dataset.shape[1]} columns."
            )

        indices = {name: column_names.index(name) for name in REQUIRED_DATA_COLUMNS}
        earth_position = read_earth_position(handle, path)
        n_rows = int(dataset.shape[0])
        if n_rows == 0:
            raise ValueError(f"No subhalos stored in {path}")

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            chunk = np.asarray(dataset[start:stop, :], dtype=np.float64)
            dgc = galactocentric_distance_kpc(
                chunk[:, indices["Xearth"]],
                chunk[:, indices["Yearth"]],
                chunk[:, indices["Zearth"]],
                earth_position,
            )
            if np.any(~np.isfinite(dgc)):
                raise ValueError(f"Non-finite reconstructed D_GC values in {path}")

            local_best = float(np.min(dgc))
            local_winners = np.flatnonzero(dgc == local_best)
            if local_best < best_dgc:
                best_dgc = local_best
                best_vmax = float(chunk[local_winners[0], indices["Vmax"]])
                n_at_best = int(local_winners.size)
            elif local_best == best_dgc:
                n_at_best += int(local_winners.size)

    if n_at_best != 1:
        raise RuntimeError(
            f"Expected one D_min subhalo in {path}; found {n_at_best}."
        )
    if not np.isfinite(best_vmax) or best_vmax <= 0.0:
        raise ValueError(
            f"D_min subhalo in {path} has invalid Vmax={best_vmax!r}."
        )
    return best_dgc, best_vmax


def make_linear_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return padded linear bin edges, including the degenerate-data case."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = 0.05 * (upper - lower)
    if padding == 0.0:
        padding = max(abs(lower), 1.0) * 1.0e-6
    return np.linspace(lower - padding, upper + padding, n_bins + 1)


def plot_vmax_distribution(
    values: np.ndarray,
    scenario: str,
    output_dir: Path,
    n_bins: int,
    dpi: int,
) -> list[Path]:
    """Draw and save the absolute-count Vmax histogram."""
    figure, axis = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    axis.hist(
        values,
        bins=make_linear_bins(values, n_bins),
        density=False,
        histtype="stepfilled",
        color=SCENARIO_COLORS[scenario],
        alpha=0.78,
    )
    axis.set_title(f"{scenario.capitalize()} scenario")
    axis.set_xlabel(r"$V_{\max}\,[\mathrm{km\ s^{-1}}]$")
    axis.set_ylabel("Number of repopulations")
    axis.set_ylim(bottom=0)
    axis.tick_params(axis="both", which="both", labelsize=15)
    axis.grid(False, which="both")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for output_format in ("png", "pdf"):
        output_path = output_dir / f"dmin_vmax_distribution.{output_format}"
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
    validate_args(args)
    csv_dmins = load_dmin_values(args.input_csv, args.scenario)
    validate_catalogue_files(args.input_root, args.scenario)

    vmax_values = []
    for sequence, repop_id in enumerate(REPOP_IDS, start=1):
        path = catalogue_path(args.input_root, repop_id, args.scenario)
        hdf5_dmin, vmax = find_dmin_subhalo_vmax(path, args.chunk_size)
        csv_dmin = csv_dmins[repop_id]
        if not np.isclose(
            hdf5_dmin,
            csv_dmin,
            rtol=args.dmin_rtol,
            atol=args.dmin_atol_kpc,
        ):
            raise RuntimeError(
                f"D_min mismatch for repop_{repop_id:04d}: HDF5="
                f"{hdf5_dmin:.17e} kpc, CSV={csv_dmin:.17e} kpc "
                f"(rtol={args.dmin_rtol:g}, atol={args.dmin_atol_kpc:g} kpc)."
            )
        vmax_values.append(vmax)
        print(
            f"[{sequence:03d}/{EXPECTED_REPOPULATIONS:03d}] "
            f"repop_{repop_id:04d} {args.scenario}: "
            f"D_min={hdf5_dmin:.8g} kpc | Vmax={vmax:.8g} km s^-1",
            flush=True,
        )

    values = np.asarray(vmax_values, dtype=np.float64)
    if values.shape != (EXPECTED_REPOPULATIONS,):
        raise RuntimeError(
            f"Expected {EXPECTED_REPOPULATIONS} Vmax values; got {values.size}."
        )
    plot_vmax_distribution(
        values=values,
        scenario=args.scenario,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
