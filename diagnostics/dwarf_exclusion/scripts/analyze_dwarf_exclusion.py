#!/usr/bin/env python3
"""Compare current and catalogue-first dwarf-exclusion brightness cuts.

This is a read-only catalogue diagnostic.  It imports the angular and
brightness-proxy definitions from ``scripts/prepare_subhalo_components.py``
and never invokes CLUMPY or reads/writes production products.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from prepare_subhalo_components import (  # noqa: E402
    build_valid_mask,
    clumpy_central_pixel_proxy_from_js,
    healpix_pixel_size_deg,
    round_up,
)


ITERATION = 0
NSIDE = 2048
ROUND_UP_DECIMALS = 2
POINTLIKE_F = 1.0e-3
EXTENDED_F = 1.0e-3
VMAX_DARK_MAX_KM_S = 8.0
DEFAULT_CHUNK_SIZE = 500_000
VALID_SCENARIOS = ("fragile", "resilient")

REQUIRED_COLUMNS = (
    "Js",
    "D_Earth",
    "Vmax",
    "theta_s",
    "r_s",
    "rho_s",
    "Xearth",
    "Yearth",
    "Zearth",
)

PER_CATALOGUE_FIELDS = (
    "scenario",
    "repop_id",
    "catalogue_path",
    "n_rows",
    "n_valid",
    "n_invalid",
    "nside",
    "theta_min_deg",
    "pointlike_f",
    "extended_f",
    "vmax_dark_max_km_s",
    "jref_original",
    "jref_dark",
    "jref_ratio_dark_to_original",
    "jref_relative_change_percent",
    "jref_changed",
    "original_ref_name",
    "original_ref_row",
    "original_ref_source_type",
    "original_ref_vmax_km_s",
    "original_ref_js",
    "original_ref_proxy",
    "original_ref_is_dwarf",
    "dark_ref_name",
    "dark_ref_row",
    "dark_ref_source_type",
    "dark_ref_vmax_km_s",
    "dark_ref_js",
    "dark_ref_proxy",
    "n_dwarfs_total",
    "n_dwarfs_pointlike",
    "n_dwarfs_extended",
    "n_dwarfs_passing_original_total",
    "n_dwarfs_passing_original_pointlike",
    "n_dwarfs_passing_original_extended",
    "n_dark_retained_a_pointlike",
    "n_dark_retained_b_pointlike",
    "retained_difference_b_minus_a_pointlike",
    "n_newly_admitted_pointlike",
    "n_a_not_b_pointlike",
    "newly_admitted_js_sum_pointlike",
    "n_dark_retained_a_extended",
    "n_dark_retained_b_extended",
    "retained_difference_b_minus_a_extended",
    "n_newly_admitted_extended",
    "n_a_not_b_extended",
    "newly_admitted_js_sum_extended",
    "n_dark_retained_a_total",
    "n_dark_retained_b_total",
    "n_newly_admitted_total",
    "n_a_not_b_total",
    "newly_admitted_js_sum_total",
    "selection_b_is_superset",
)


@dataclass(frozen=True)
class ReferenceObject:
    proxy: float
    row: int
    name: str
    source_type: str
    vmax: float
    js: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root containing repop_XXXX catalogue directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or resumable diagnostic-only output directory.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--repop-id", type=int, help="Process one repopulation.")
    selection.add_argument(
        "--repop-start",
        type=int,
        help="First repopulation in an inclusive range (default: 0).",
    )
    parser.add_argument(
        "--repop-end",
        type=int,
        help="Last repopulation in the inclusive range (default: 499).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=VALID_SCENARIOS,
        default=list(VALID_SCENARIOS),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="HDF5 rows read at once.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated rows already present in per_catalogue.csv.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Report and skip missing catalogues instead of stopping.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV outputs but do not generate plots.",
    )
    return parser.parse_args()


def catalogue_path(input_root: Path, repop_id: int, scenario: str) -> Path:
    """Use the same scenario-specific convention as the production pipeline."""
    return (
        Path(input_root)
        / f"repop_{repop_id:04d}"
        / f"fullrepop_hydro_{scenario}.h5"
    )


def decode_strings(values: Iterable[object]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def validate_unit(raw: object, expected: u.UnitBase, column: str, path: Path) -> None:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    try:
        actual = u.Unit(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid unit for {column!r} in {path}: {text!r}") from error
    if not actual.is_equivalent(expected):
        raise ValueError(
            f"Unexpected unit for {column!r} in {path}: {text!r}; "
            f"expected a unit equivalent to {expected}"
        )


def open_catalogue(path: Path):
    """Open and validate the production HDF5 structure and column metadata."""
    handle = h5py.File(path, "r")
    try:
        group_name = f"iteration_{ITERATION}"
        if group_name not in handle:
            raise KeyError(f"Missing HDF5 group {group_name!r} in {path}")
        group = handle[group_name]
        if "data" not in group or "halo_name" not in group:
            raise KeyError(f"Missing {group_name}/data or halo_name in {path}")
        data = group["data"]
        names = group["halo_name"]
        if data.ndim != 2:
            raise ValueError(f"Expected a two-dimensional data table in {path}")
        if names.ndim != 1 or names.shape[0] != data.shape[0]:
            raise ValueError(f"halo_name/data row mismatch in {path}")
        if "column_names" not in data.attrs:
            raise KeyError(f"Missing data column_names metadata in {path}")
        columns = decode_strings(data.attrs["column_names"])
        if len(columns) != data.shape[1] or len(set(columns)) != len(columns):
            raise ValueError(f"Invalid or duplicate column_names metadata in {path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise KeyError(f"Missing required column(s) in {path}: {', '.join(missing)}")
        indices = {column: columns.index(column) for column in REQUIRED_COLUMNS}
        if "units" in data.attrs:
            units = list(data.attrs["units"])
            if len(units) != data.shape[1]:
                raise ValueError(f"units metadata length mismatch in {path}")
            validate_unit(units[indices["Js"]], u.GeV**2 / u.cm**5, "Js", path)
            validate_unit(units[indices["Vmax"]], u.km / u.s, "Vmax", path)
        return handle, data, names, indices
    except Exception:
        handle.close()
        raise


def chunk_arrays(array: np.ndarray, indices: dict[str, int]) -> dict[str, np.ndarray]:
    return {
        column: np.asarray(array[:, index], dtype=np.float64)
        for column, index in indices.items()
    }


def valid_mask(values: dict[str, np.ndarray]) -> np.ndarray:
    return build_valid_mask(
        js=values["Js"],
        d_earth=values["D_Earth"],
        theta_s=values["theta_s"],
        r_s=values["r_s"],
        rho_s=values["rho_s"],
        x_e=values["Xearth"],
        y_e=values["Yearth"],
        z_e=values["Zearth"],
    )


def masks_and_proxy(
    values: dict[str, np.ndarray], valid: np.ndarray, theta_min_deg: float, nside: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pointlike = valid & (values["theta_s"] < theta_min_deg)
    extended = valid & (values["theta_s"] >= theta_min_deg)
    proxy = np.zeros(valid.size, dtype=np.float64)
    proxy[pointlike] = values["Js"][pointlike]
    if np.any(extended):
        proxy[extended] = clumpy_central_pixel_proxy_from_js(
            js=values["Js"][extended],
            d_earth_kpc=values["D_Earth"][extended],
            rs_kpc=values["r_s"][extended],
            nside=nside,
        )
    return pointlike, extended, proxy


def best_reference(
    current: ReferenceObject | None,
    candidate_mask: np.ndarray,
    proxy: np.ndarray,
    values: dict[str, np.ndarray],
    names: list[str],
    row_start: int,
    pointlike: np.ndarray,
) -> ReferenceObject | None:
    indices = np.flatnonzero(candidate_mask)
    if indices.size == 0:
        return current
    local = int(indices[np.argmax(proxy[indices])])
    candidate = ReferenceObject(
        proxy=float(proxy[local]),
        row=row_start + local,
        name=names[local],
        source_type="pointlike" if pointlike[local] else "extended",
        vmax=float(values["Vmax"][local]),
        js=float(values["Js"][local]),
    )
    if current is None or candidate.proxy > current.proxy:
        return candidate
    if candidate.proxy == current.proxy and candidate.row < current.row:
        return candidate
    return current


def scan_catalogue(
    path: Path,
    repop_id: int,
    scenario: str,
    *,
    nside: int = NSIDE,
    pointlike_f: float = POINTLIKE_F,
    extended_f: float = EXTENDED_F,
    vmax_dark_max: float = VMAX_DARK_MAX_KM_S,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: bool = True,
) -> dict[str, object]:
    """Perform the two exact selection branches with two chunked HDF5 passes."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of two")
    if pointlike_f < 0.0 or extended_f < 0.0:
        raise ValueError("cut fractions must be non-negative")

    theta_min_deg = round_up(
        healpix_pixel_size_deg(nside), decimals=ROUND_UP_DECIMALS
    )
    original_ref: ReferenceObject | None = None
    dark_ref: ReferenceObject | None = None
    counts = {
        "n_valid": 0,
        "n_dwarfs_total": 0,
        "n_dwarfs_pointlike": 0,
        "n_dwarfs_extended": 0,
    }

    handle, data, name_data, indices = open_catalogue(path)
    try:
        n_rows = int(data.shape[0])
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            values = chunk_arrays(data[start:stop], indices)
            names = decode_strings(name_data[start:stop])
            valid = valid_mask(values)
            bad_vmax = valid & (~np.isfinite(values["Vmax"]) | (values["Vmax"] <= 0.0))
            if np.any(bad_vmax):
                row = start + int(np.flatnonzero(bad_vmax)[0])
                raise ValueError(f"Invalid Vmax for otherwise-valid halo at row {row} in {path}")
            pointlike, extended, proxy = masks_and_proxy(
                values, valid, theta_min_deg, nside
            )
            dark = valid & (values["Vmax"] <= vmax_dark_max)
            dwarf = valid & (values["Vmax"] > vmax_dark_max)
            original_ref = best_reference(
                original_ref, valid, proxy, values, names, start, pointlike
            )
            dark_ref = best_reference(
                dark_ref, dark, proxy, values, names, start, pointlike
            )
            counts["n_valid"] += int(np.count_nonzero(valid))
            counts["n_dwarfs_total"] += int(np.count_nonzero(dwarf))
            counts["n_dwarfs_pointlike"] += int(np.count_nonzero(dwarf & pointlike))
            counts["n_dwarfs_extended"] += int(np.count_nonzero(dwarf & extended))
            if progress:
                print(
                    f"{scenario} repop_{repop_id:04d} reference pass: "
                    f"{stop:,}/{n_rows:,}",
                    flush=True,
                )

        if original_ref is None or original_ref.proxy <= 0.0:
            raise RuntimeError(f"No positive valid reference object in {path}")
        if dark_ref is None or dark_ref.proxy <= 0.0:
            raise RuntimeError(f"No positive valid dark reference object in {path}")
        if dark_ref.proxy > original_ref.proxy:
            raise AssertionError("Jref_dark exceeds Jref_original")

        metrics = {
            "n_dwarfs_passing_original_pointlike": 0,
            "n_dwarfs_passing_original_extended": 0,
            "n_dark_retained_a_pointlike": 0,
            "n_dark_retained_b_pointlike": 0,
            "n_newly_admitted_pointlike": 0,
            "n_a_not_b_pointlike": 0,
            "newly_admitted_js_sum_pointlike": 0.0,
            "n_dark_retained_a_extended": 0,
            "n_dark_retained_b_extended": 0,
            "n_newly_admitted_extended": 0,
            "n_a_not_b_extended": 0,
            "newly_admitted_js_sum_extended": 0.0,
        }
        original_point_cut = pointlike_f * original_ref.proxy
        original_extended_cut = extended_f * original_ref.proxy
        dark_point_cut = pointlike_f * dark_ref.proxy
        dark_extended_cut = extended_f * dark_ref.proxy

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            values = chunk_arrays(data[start:stop], indices)
            valid = valid_mask(values)
            pointlike, extended, proxy = masks_and_proxy(
                values, valid, theta_min_deg, nside
            )
            dark = valid & (values["Vmax"] <= vmax_dark_max)
            dwarf = valid & (values["Vmax"] > vmax_dark_max)
            pass_a_point = pointlike & (values["Js"] >= original_point_cut)
            pass_b_point = pointlike & (values["Js"] >= dark_point_cut)
            pass_a_ext = extended & (proxy >= original_extended_cut)
            pass_b_ext = extended & (proxy >= dark_extended_cut)

            for source, source_mask, pass_a, pass_b in (
                ("pointlike", pointlike, pass_a_point, pass_b_point),
                ("extended", extended, pass_a_ext, pass_b_ext),
            ):
                retained_a = dark & source_mask & pass_a
                retained_b = dark & source_mask & pass_b
                newly = retained_b & ~retained_a
                lost = retained_a & ~retained_b
                metrics[f"n_dwarfs_passing_original_{source}"] += int(
                    np.count_nonzero(dwarf & source_mask & pass_a)
                )
                metrics[f"n_dark_retained_a_{source}"] += int(np.count_nonzero(retained_a))
                metrics[f"n_dark_retained_b_{source}"] += int(np.count_nonzero(retained_b))
                metrics[f"n_newly_admitted_{source}"] += int(np.count_nonzero(newly))
                metrics[f"n_a_not_b_{source}"] += int(np.count_nonzero(lost))
                metrics[f"newly_admitted_js_sum_{source}"] += float(
                    values["Js"][newly].sum(dtype=np.float64)
                )
                if np.any(newly):
                    relevant_cut = dark_point_cut if source == "pointlike" else dark_extended_cut
                    relevant_brightness = values["Js"] if source == "pointlike" else proxy
                    if np.any(relevant_brightness[newly] < relevant_cut):
                        raise AssertionError("Newly admitted object fails dark-only threshold")
            if progress:
                print(
                    f"{scenario} repop_{repop_id:04d} selection pass: "
                    f"{stop:,}/{n_rows:,}",
                    flush=True,
                )
    finally:
        handle.close()

    ratio = dark_ref.proxy / original_ref.proxy
    row: dict[str, object] = {
        "scenario": scenario,
        "repop_id": repop_id,
        "catalogue_path": str(path),
        "n_rows": n_rows,
        "n_valid": counts["n_valid"],
        "n_invalid": n_rows - counts["n_valid"],
        "nside": nside,
        "theta_min_deg": theta_min_deg,
        "pointlike_f": pointlike_f,
        "extended_f": extended_f,
        "vmax_dark_max_km_s": vmax_dark_max,
        "jref_original": original_ref.proxy,
        "jref_dark": dark_ref.proxy,
        "jref_ratio_dark_to_original": ratio,
        "jref_relative_change_percent": 100.0 * (ratio - 1.0),
        "jref_changed": dark_ref.proxy < original_ref.proxy,
        "original_ref_name": original_ref.name,
        "original_ref_row": original_ref.row,
        "original_ref_source_type": original_ref.source_type,
        "original_ref_vmax_km_s": original_ref.vmax,
        "original_ref_js": original_ref.js,
        "original_ref_proxy": original_ref.proxy,
        "original_ref_is_dwarf": original_ref.vmax > vmax_dark_max,
        "dark_ref_name": dark_ref.name,
        "dark_ref_row": dark_ref.row,
        "dark_ref_source_type": dark_ref.source_type,
        "dark_ref_vmax_km_s": dark_ref.vmax,
        "dark_ref_js": dark_ref.js,
        "dark_ref_proxy": dark_ref.proxy,
        **counts,
        **metrics,
    }
    row["n_dwarfs_passing_original_total"] = (
        row["n_dwarfs_passing_original_pointlike"]
        + row["n_dwarfs_passing_original_extended"]
    )
    for source in ("pointlike", "extended"):
        row[f"retained_difference_b_minus_a_{source}"] = (
            row[f"n_dark_retained_b_{source}"] - row[f"n_dark_retained_a_{source}"]
        )
    for selection in ("a", "b"):
        row[f"n_dark_retained_{selection}_total"] = sum(
            row[f"n_dark_retained_{selection}_{source}"]
            for source in ("pointlike", "extended")
        )
    row["n_newly_admitted_total"] = sum(
        row[f"n_newly_admitted_{source}"] for source in ("pointlike", "extended")
    )
    row["n_a_not_b_total"] = sum(
        row[f"n_a_not_b_{source}"] for source in ("pointlike", "extended")
    )
    row["newly_admitted_js_sum_total"] = sum(
        row[f"newly_admitted_js_sum_{source}"]
        for source in ("pointlike", "extended")
    )
    row["selection_b_is_superset"] = row["n_a_not_b_total"] == 0

    if counts["n_dwarfs_total"] != (
        counts["n_dwarfs_pointlike"] + counts["n_dwarfs_extended"]
    ):
        raise AssertionError("Dwarf classification counts are inconsistent")
    if row["n_newly_admitted_total"] != (
        row["n_dark_retained_b_total"] - row["n_dark_retained_a_total"]
    ):
        raise AssertionError("Retained-set difference counts are inconsistent")
    if not row["selection_b_is_superset"]:
        raise AssertionError("Dark-only selection is not a superset of original dark selection")
    return row


def csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.17e}"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return value


def write_per_catalogue(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PER_CATALOGUE_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: csv_value(row[field]) for field in PER_CATALOGUE_FIELDS}
            for row in rows
        )


def load_existing(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PER_CATALOGUE_FIELDS:
            raise ValueError(f"Unexpected columns in resumable CSV {path}")
        return list(reader)


def numeric(rows: list[dict[str, object]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def truth(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for scenario in VALID_SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        if not selected:
            continue
        ratio = numeric(selected, "jref_ratio_dark_to_original")
        point = numeric(selected, "n_newly_admitted_pointlike")
        extended = numeric(selected, "n_newly_admitted_extended")
        js_point = numeric(selected, "newly_admitted_js_sum_pointlike")
        js_extended = numeric(selected, "newly_admitted_js_sum_extended")
        summary: dict[str, object] = {
            "scenario": scenario,
            "n_repops": len(selected),
            "fraction_jref_changed": np.mean([truth(row["jref_changed"]) for row in selected]),
            "fraction_original_reference_is_dwarf": np.mean(
                [truth(row["original_ref_is_dwarf"]) for row in selected]
            ),
        }
        for name, values in (("jref_ratio", ratio),):
            summary.update(
                {
                    f"{name}_mean": np.mean(values),
                    f"{name}_median": np.median(values),
                    f"{name}_min": np.min(values),
                    f"{name}_p05": np.percentile(values, 5),
                    f"{name}_p16": np.percentile(values, 16),
                    f"{name}_p25": np.percentile(values, 25),
                    f"{name}_p75": np.percentile(values, 75),
                    f"{name}_p84": np.percentile(values, 84),
                    f"{name}_p95": np.percentile(values, 95),
                    f"{name}_max": np.max(values),
                }
            )
        for source, counts_array, js_array in (
            ("pointlike", point, js_point),
            ("extended", extended, js_extended),
        ):
            summary.update(
                {
                    f"newly_admitted_{source}_mean": np.mean(counts_array),
                    f"newly_admitted_{source}_median": np.median(counts_array),
                    f"newly_admitted_{source}_min": np.min(counts_array),
                    f"newly_admitted_{source}_max": np.max(counts_array),
                    f"newly_admitted_{source}_js_total": np.sum(js_array),
                    f"newly_admitted_{source}_js_average_per_repop": np.mean(js_array),
                }
            )
        combined_js = js_point + js_extended
        summary["newly_admitted_total_js_total"] = np.sum(combined_js)
        summary["newly_admitted_total_js_average_per_repop"] = np.mean(combined_js)
        summaries.append(summary)
    return summaries


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: csv_value(row[field]) for field in fields} for row in rows
        )


def plot_results(rows: list[dict[str, object]], output_dir: Path) -> None:
    colors = {"fragile": "#d95f02", "resilient": "#1b9e77"}
    fig, axis = plt.subplots(figsize=(7, 5))
    for scenario in VALID_SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        if selected:
            axis.hist(
                numeric(selected, "jref_ratio_dark_to_original"),
                bins=30,
                alpha=0.55,
                label=scenario,
                color=colors[scenario],
            )
    axis.set_xlabel(r"$J_{\rm ref,dark}/J_{\rm ref,original}$")
    axis.set_ylabel("Repopulations")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "jref_ratio_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for scenario in VALID_SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        if selected:
            axis.hist(
                numeric(selected, "n_newly_admitted_extended"),
                bins=30,
                alpha=0.55,
                label=scenario,
                color=colors[scenario],
            )
    axis.set_xlabel("Newly admitted extended dark subhalos")
    axis.set_ylabel("Repopulations")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "newly_admitted_extended_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for scenario in VALID_SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        if selected:
            axis.scatter(
                numeric(selected, "jref_ratio_dark_to_original"),
                numeric(selected, "n_newly_admitted_extended"),
                s=18,
                alpha=0.7,
                label=scenario,
                color=colors[scenario],
            )
    axis.set_xlabel(r"$J_{\rm ref,dark}/J_{\rm ref,original}$")
    axis.set_ylabel("Newly admitted extended dark subhalos")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "jref_ratio_vs_new_extended.png", dpi=180)
    plt.close(fig)


def requested_repop_ids(args: argparse.Namespace) -> range:
    if args.repop_id is not None:
        if args.repop_id < 0:
            raise ValueError("--repop-id must be non-negative")
        if args.repop_end is not None:
            raise ValueError("--repop-end cannot be combined with --repop-id")
        return range(args.repop_id, args.repop_id + 1)
    start = 0 if args.repop_start is None else args.repop_start
    end = 499 if args.repop_end is None else args.repop_end
    if start < 0 or end < start:
        raise ValueError("Require 0 <= repop-start <= repop-end")
    return range(start, end + 1)


def main() -> None:
    args = parse_args()
    repop_ids = requested_repop_ids(args)
    output_dir = args.output_dir.resolve()
    input_root = args.input_root.resolve()
    if output_dir == input_root or input_root in output_dir.parents:
        raise ValueError("Diagnostic output directory must be outside the input catalogue tree")
    per_catalogue_path = output_dir / "per_catalogue.csv"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}; use --resume only for this diagnostic"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_existing(per_catalogue_path) if args.resume else []
    completed = {(row["scenario"], int(row["repop_id"])) for row in rows}

    for scenario in args.scenarios:
        for repop_id in repop_ids:
            key = (scenario, repop_id)
            if key in completed:
                print(f"Skipping completed {scenario} repop_{repop_id:04d}", flush=True)
                continue
            path = catalogue_path(input_root, repop_id, scenario)
            if not path.is_file():
                message = f"Missing catalogue: {path}"
                if args.skip_missing:
                    print(f"WARNING: {message}", file=sys.stderr, flush=True)
                    continue
                raise FileNotFoundError(message)
            row = scan_catalogue(
                path,
                repop_id,
                scenario,
                chunk_size=args.chunk_size,
                progress=True,
            )
            rows.append(row)
            rows.sort(key=lambda item: (str(item["scenario"]), int(item["repop_id"])))
            write_per_catalogue(per_catalogue_path, rows)
            completed.add(key)
            print(
                f"Completed {scenario} repop_{repop_id:04d}: "
                f"ratio={float(row['jref_ratio_dark_to_original']):.8g}, "
                f"new pointlike={row['n_newly_admitted_pointlike']}, "
                f"new extended={row['n_newly_admitted_extended']}",
                flush=True,
            )

    summaries = summarize(rows)
    write_dict_csv(output_dir / "summary_by_scenario.csv", summaries)
    if rows and not args.no_plots:
        plot_results(rows, output_dir)
    print(f"Saved per-catalogue results: {per_catalogue_path}")
    print(f"Saved scenario summary: {output_dir / 'summary_by_scenario.csv'}")


if __name__ == "__main__":
    main()
