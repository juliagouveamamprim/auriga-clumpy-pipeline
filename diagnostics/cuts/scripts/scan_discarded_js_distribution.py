#!/usr/bin/env python3
"""Build per-catalogue histograms of discarded-subhalo relative Js."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from diagnostics.cuts.scripts.scan_central_pixel_cuts import (  # noqa: E402
    apply_vmax_selection,
    selection_tag,
    validate_nside,
)
from diagnostics.cuts.scripts.scan_relative_js_cumulative import (  # noqa: E402
    LOG_REL_MAX,
    LOG_REL_MIN,
    N_BINS_REL,
    VALID_SCENARIOS,
    catalogue_path,
    parse_scenarios,
)
from scripts.prepare_subhalo_components import (  # noqa: E402
    CHUNK_SIZE,
    ITERATION,
    ROUND_UP_DECIMALS,
    build_valid_mask,
    clumpy_central_pixel_proxy_from_js,
    compute_pixel_reference,
    healpix_pixel_size_deg,
    round_up,
)


FORMAT_VERSION = 2
POPULATIONS = ("all", "pointlike", "extended")
DEFAULT_F = 1.0e-3
DEFAULT_NSIDE = 2048

REQUIRED_H5_COLUMNS = (
    "Js",
    "D_Earth",
    "theta_s",
    "r_s",
    "rho_s",
    "Xearth",
    "Yearth",
    "Zearth",
)

POPULATION_SCALAR_FIELDS = (
    "n_discarded",
    "n_underflow",
    "n_overflow",
    "n_invalid_j_rel",
    "max_js_discarded",
    "max_j_rel_discarded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one binned Js/Js,max distribution for the subhalos "
            "discarded by the catalogue cut in each requested catalogue."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Directory containing repop_XXXX/fullrepop_hydro_"
            "<scenario>.h5 files."
        ),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        required=True,
        help="Small checkpointable NPZ containing per-catalogue histograms.",
    )
    parser.add_argument("--repop-start", type=int, default=0)
    parser.add_argument("--n-repops", type=int, default=500)
    parser.add_argument(
        "--scenarios",
        type=parse_scenarios,
        default=VALID_SCENARIOS,
        help="Comma-separated scenarios (default: fragile,resilient).",
    )
    parser.add_argument("--nside", type=int, default=DEFAULT_NSIDE)
    parser.add_argument("--f", type=float, default=DEFAULT_F)
    parser.add_argument(
        "--max-vmax-km-s",
        type=float,
        default=None,
        help=(
            "Optional inclusive Vmax ceiling in km/s, applied after the "
            "production validity mask and before Js,max, J_pixel_ref, "
            "morphology, and brightness cuts."
        ),
    )
    parser.add_argument("--log-rel-min", type=float, default=LOG_REL_MIN)
    parser.add_argument("--log-rel-max", type=float, default=LOG_REL_MAX)
    parser.add_argument("--n-bins", type=int, default=N_BINS_REL)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repop_start < 0:
        raise ValueError("--repop-start must be non-negative.")
    if args.n_repops <= 0:
        raise ValueError("--n-repops must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive.")
    if not np.isfinite(args.f) or args.f < 0.0:
        raise ValueError("--f must be finite and non-negative.")
    selection_tag(args.max_vmax_km_s)
    if (
        not np.isfinite(args.log_rel_min)
        or not np.isfinite(args.log_rel_max)
        or args.log_rel_min >= args.log_rel_max
    ):
        raise ValueError(
            "Relative-J limits must be finite and log-rel-min < log-rel-max."
        )
    if args.output_npz.suffix != ".npz":
        raise ValueError("--output-npz must end in .npz.")
    validate_nside(args.nside)


def eligible_mask(
    array: np.ndarray,
    column_indices: dict[str, int],
    max_vmax_km_s: float | None,
    row_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return production-valid and physically selected masks for one chunk."""
    valid = build_valid_mask(
        js=array[:, column_indices["Js"]],
        d_earth=array[:, column_indices["D_Earth"]],
        theta_s=array[:, column_indices["theta_s"]],
        r_s=array[:, column_indices["r_s"]],
        rho_s=array[:, column_indices["rho_s"]],
        x_e=array[:, column_indices["Xearth"]],
        y_e=array[:, column_indices["Yearth"]],
        z_e=array[:, column_indices["Zearth"]],
    )
    physical_valid = valid.copy()
    if max_vmax_km_s is None:
        return physical_valid, valid, 0
    return (
        physical_valid,
        *apply_vmax_selection(
            valid,
            array[:, column_indices["Vmax"]],
            max_vmax_km_s,
            row_offset=row_offset,
        ),
    )


def empty_population(n_bins: int) -> dict[str, object]:
    return {
        "histogram": np.zeros(n_bins, dtype=np.int64),
        "n_discarded": 0,
        "n_underflow": 0,
        "n_overflow": 0,
        "n_invalid_j_rel": 0,
        "max_js_discarded": float("nan"),
        "max_j_rel_discarded": float("nan"),
    }


def accumulate_discarded(
    result: dict[str, object],
    js: np.ndarray,
    js_max_cat: float,
    log_relative_edges: np.ndarray,
) -> None:
    """Accumulate one population without retaining halo-level values."""
    if not np.isfinite(js_max_cat) or js_max_cat <= 0.0:
        raise ValueError("js_max_cat must be finite and positive.")

    js = np.asarray(js, dtype=np.float64)
    result["n_discarded"] += int(js.size)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        j_rel = js / js_max_cat
        log_j_rel = np.log10(j_rel)

    valid = np.isfinite(log_j_rel) & (j_rel > 0.0)
    result["n_invalid_j_rel"] += int(np.count_nonzero(~valid))

    if not np.any(valid):
        return

    valid_js = js[valid]
    valid_j_rel = j_rel[valid]
    valid_log_j_rel = log_j_rel[valid]

    previous_max_js = float(result["max_js_discarded"])
    previous_max_rel = float(result["max_j_rel_discarded"])
    local_max_js = float(np.max(valid_js))
    local_max_rel = float(np.max(valid_j_rel))
    result["max_js_discarded"] = (
        local_max_js
        if not np.isfinite(previous_max_js)
        else max(previous_max_js, local_max_js)
    )
    result["max_j_rel_discarded"] = (
        local_max_rel
        if not np.isfinite(previous_max_rel)
        else max(previous_max_rel, local_max_rel)
    )

    underflow = valid_log_j_rel < log_relative_edges[0]
    overflow = valid_log_j_rel > log_relative_edges[-1]
    in_range = ~(underflow | overflow)

    result["n_underflow"] += int(np.count_nonzero(underflow))
    result["n_overflow"] += int(np.count_nonzero(overflow))
    result["histogram"] += np.histogram(
        valid_log_j_rel[in_range],
        bins=log_relative_edges,
    )[0]


def combine_populations(
    pointlike: dict[str, object],
    extended: dict[str, object],
) -> dict[str, object]:
    combined = empty_population(len(pointlike["histogram"]))
    combined["histogram"] = (
        pointlike["histogram"] + extended["histogram"]
    )

    for field in (
        "n_discarded",
        "n_underflow",
        "n_overflow",
        "n_invalid_j_rel",
    ):
        combined[field] = int(pointlike[field]) + int(extended[field])

    for field in ("max_js_discarded", "max_j_rel_discarded"):
        values = [
            float(population[field])
            for population in (pointlike, extended)
            if np.isfinite(float(population[field]))
        ]
        combined[field] = max(values) if values else float("nan")

    return combined


def classify_discarded(
    array: np.ndarray,
    column_indices: dict[str, int],
    theta_min_deg: float,
    nside: int,
    j_cut: float,
    max_vmax_km_s: float | None = None,
    row_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the production validity, geometry, and cut conventions."""
    js = array[:, column_indices["Js"]]
    d_earth = array[:, column_indices["D_Earth"]]
    theta_s = array[:, column_indices["theta_s"]]
    r_s = array[:, column_indices["r_s"]]
    _, valid, _ = eligible_mask(
        array,
        column_indices,
        max_vmax_km_s,
        row_offset=row_offset,
    )
    pointlike = valid & (theta_s < theta_min_deg)
    extended = valid & (theta_s >= theta_min_deg)
    pointlike_discarded = pointlike & (js < j_cut)
    extended_discarded = np.zeros(len(array), dtype=bool)

    if np.any(extended):
        extended_proxy = clumpy_central_pixel_proxy_from_js(
            js=js[extended],
            d_earth_kpc=d_earth[extended],
            rs_kpc=r_s[extended],
            nside=nside,
        )
        extended_indices = np.flatnonzero(extended)
        extended_discarded[extended_indices] = extended_proxy < j_cut

    return valid, pointlike_discarded, extended_discarded


def read_catalogue_table(
    handle: h5py.File,
    path: Path,
    max_vmax_km_s: float | None = None,
):
    group_name = f"iteration_{ITERATION}"
    dataset_name = f"{group_name}/data"

    if dataset_name not in handle:
        raise KeyError(f"Missing dataset {dataset_name!r} in {path}")

    data = handle[dataset_name]
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D table in {path}; got {data.shape}.")
    if "column_names" not in data.attrs:
        raise KeyError(f"Dataset {dataset_name!r} has no column_names in {path}")

    column_names = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in data.attrs["column_names"]
    ]
    required_columns = list(REQUIRED_H5_COLUMNS)
    if max_vmax_km_s is not None:
        required_columns.append("Vmax")
    missing = [name for name in required_columns if name not in column_names]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {', '.join(missing)}")

    column_indices = {
        name: column_names.index(name)
        for name in required_columns
    }
    return data, column_indices


def scan_selected_js_max(
    data,
    column_indices: dict[str, int],
    n_rows: int,
    chunk_size: int,
    max_vmax_km_s: float | None,
) -> dict[str, object]:
    """Find Js,max from the valid selected population before brightness cuts."""
    js_max = -np.inf
    n_physical_valid = 0
    n_selected = 0
    n_vmax_excluded = 0

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        array = data[start:stop]
        physical_valid, selected, n_excluded = eligible_mask(
            array,
            column_indices,
            max_vmax_km_s,
            row_offset=start,
        )
        n_physical_valid += int(np.count_nonzero(physical_valid))
        n_selected += int(np.count_nonzero(selected))
        n_vmax_excluded += n_excluded
        if np.any(selected):
            js_max = max(
                js_max,
                float(np.max(array[selected, column_indices["Js"]])),
            )

    if not np.isfinite(js_max) or js_max <= 0.0:
        raise RuntimeError("No valid subhalo remains in the selected population.")
    return {
        "js_max_cat": js_max,
        "n_physical_valid_rows": n_physical_valid,
        "n_selected_rows": n_selected,
        "n_vmax_excluded": n_vmax_excluded,
    }


def process_catalogue(
    path: Path,
    nside: int,
    f_value: float,
    chunk_size: int,
    log_relative_edges: np.ndarray,
    max_vmax_km_s: float | None = None,
) -> dict[str, object]:
    theta_min_deg = round_up(
        healpix_pixel_size_deg(nside),
        decimals=ROUND_UP_DECIMALS,
    )

    with h5py.File(path, "r") as handle:
        data, column_indices = read_catalogue_table(
            handle, path, max_vmax_km_s
        )
        n_rows = int(data.shape[0])
        selection_info = scan_selected_js_max(
            data=data,
            column_indices=column_indices,
            n_rows=n_rows,
            chunk_size=chunk_size,
            max_vmax_km_s=max_vmax_km_s,
        )
        js_max_cat = float(selection_info["js_max_cat"])
        ref_info = compute_pixel_reference(
            data=data,
            column_indices=column_indices,
            n_total=n_rows,
            theta_min_deg=theta_min_deg,
            nside=nside,
            chunk_size=chunk_size,
            max_vmax_km_s=max_vmax_km_s,
        )
        j_pixel_ref = float(ref_info["j_pixel_ref"])
        j_cut = f_value * j_pixel_ref
        populations = {
            "pointlike": empty_population(len(log_relative_edges) - 1),
            "extended": empty_population(len(log_relative_edges) - 1),
        }
        n_valid_rows = 0

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            array = data[start:stop]
            valid, pointlike_discarded, extended_discarded = (
                classify_discarded(
                    array=array,
                    column_indices=column_indices,
                    theta_min_deg=theta_min_deg,
                    nside=nside,
                    j_cut=j_cut,
                    max_vmax_km_s=max_vmax_km_s,
                    row_offset=start,
                )
            )
            n_valid_rows += int(np.count_nonzero(valid))
            js = array[:, column_indices["Js"]]
            accumulate_discarded(
                populations["pointlike"],
                js[pointlike_discarded],
                js_max_cat,
                log_relative_edges,
            )
            accumulate_discarded(
                populations["extended"],
                js[extended_discarded],
                js_max_cat,
                log_relative_edges,
            )

        if n_valid_rows != selection_info["n_selected_rows"]:
            raise RuntimeError(
                f"Selected-population count mismatch for {path}: "
                f"max scan={selection_info['n_selected_rows']}, "
                f"cut scan={n_valid_rows}."
            )

    populations["all"] = combine_populations(
        populations["pointlike"],
        populations["extended"],
    )

    for population, population_result in populations.items():
        counted = (
            int(np.sum(population_result["histogram"], dtype=np.int64))
            + int(population_result["n_underflow"])
            + int(population_result["n_overflow"])
            + int(population_result["n_invalid_j_rel"])
        )
        if counted != population_result["n_discarded"]:
            raise RuntimeError(
                f"Discarded-count mismatch for {population} in {path}: "
                f"counted {counted}, expected "
                f"{population_result['n_discarded']}."
            )

    result: dict[str, object] = {
        "js_max_cat": js_max_cat,
        "j_pixel_ref": j_pixel_ref,
        "n_input_rows": n_rows,
        "n_valid_rows": n_valid_rows,
        "n_physical_valid_rows": selection_info["n_physical_valid_rows"],
        "n_invalid_input_rows": (
            n_rows - int(selection_info["n_physical_valid_rows"])
        ),
        "n_vmax_excluded": selection_info["n_vmax_excluded"],
    }
    for population in POPULATIONS:
        population_result = populations[population]
        result[f"hist_{population}"] = population_result["histogram"]
        for field in POPULATION_SCALAR_FIELDS:
            result[f"{field}_{population}"] = population_result[field]

    return result


def checkpoint_metadata(
    args: argparse.Namespace,
    repop_ids: np.ndarray,
    log_relative_edges: np.ndarray,
) -> dict[str, object]:
    max_vmax_km_s = getattr(args, "max_vmax_km_s", None)
    return {
        "f": float(args.f),
        "nside": int(args.nside),
        "theta_min_deg": round_up(
            healpix_pixel_size_deg(args.nside),
            decimals=ROUND_UP_DECIMALS,
        ),
        "log_relative_edges": np.asarray(
            log_relative_edges,
            dtype=np.float64,
        ),
        "requested_repop_ids": np.asarray(repop_ids, dtype=np.int64),
        "requested_scenarios": tuple(args.scenarios),
        "input_root": str(args.input_root.resolve()),
        "population_selection": selection_tag(max_vmax_km_s),
        "max_vmax_km_s": (
            float("nan")
            if max_vmax_km_s is None
            else float(max_vmax_km_s)
        ),
    }


def checkpoint_payload(
    metadata: dict[str, object],
    entries: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    n_bins = len(metadata["log_relative_edges"]) - 1
    payload = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
        "f": np.asarray(metadata["f"], dtype=np.float64),
        "nside": np.asarray(metadata["nside"], dtype=np.int64),
        "theta_min_deg": np.asarray(
            metadata["theta_min_deg"], dtype=np.float64
        ),
        "log_relative_edges": np.asarray(
            metadata["log_relative_edges"], dtype=np.float64
        ),
        "requested_repop_ids": np.asarray(
            metadata["requested_repop_ids"], dtype=np.int64
        ),
        "requested_scenarios": np.asarray(
            metadata["requested_scenarios"], dtype=np.str_
        ),
        "input_root": np.asarray(metadata["input_root"], dtype=np.str_),
        "population_selection": np.asarray(
            metadata["population_selection"], dtype=np.str_
        ),
        "max_vmax_km_s": np.asarray(
            metadata["max_vmax_km_s"], dtype=np.float64
        ),
        "catalogue_repop_ids": np.asarray(
            [entry["repop_id"] for entry in entries], dtype=np.int64
        ),
        "catalogue_scenarios": np.asarray(
            [entry["scenario"] for entry in entries], dtype=np.str_
        ),
        "source_h5": np.asarray(
            [entry["source_h5"] for entry in entries], dtype=np.str_
        ),
        "js_max_cat": np.asarray(
            [entry["js_max_cat"] for entry in entries], dtype=np.float64
        ),
        "j_pixel_ref": np.asarray(
            [entry["j_pixel_ref"] for entry in entries], dtype=np.float64
        ),
        "n_input_rows": np.asarray(
            [entry["n_input_rows"] for entry in entries], dtype=np.int64
        ),
        "n_valid_rows": np.asarray(
            [entry["n_valid_rows"] for entry in entries], dtype=np.int64
        ),
        "n_physical_valid_rows": np.asarray(
            [entry["n_physical_valid_rows"] for entry in entries],
            dtype=np.int64,
        ),
        "n_vmax_excluded": np.asarray(
            [entry["n_vmax_excluded"] for entry in entries], dtype=np.int64
        ),
        "n_invalid_input_rows": np.asarray(
            [entry["n_invalid_input_rows"] for entry in entries],
            dtype=np.int64,
        ),
    }

    for population in POPULATIONS:
        payload[f"hist_{population}"] = (
            np.stack([entry[f"hist_{population}"] for entry in entries])
            if entries
            else np.empty((0, n_bins), dtype=np.int64)
        )
        for field in POPULATION_SCALAR_FIELDS:
            dtype = (
                np.int64
                if field.startswith("n_")
                else np.float64
            )
            payload[f"{field}_{population}"] = np.asarray(
                [entry[f"{field}_{population}"] for entry in entries],
                dtype=dtype,
            )

    return payload


def save_checkpoint(
    path: Path,
    metadata: dict[str, object],
    entries: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **checkpoint_payload(metadata, entries))

    temporary.replace(path)


def load_checkpoint(
    path: Path,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    if not path.exists():
        return []

    with np.load(path, allow_pickle=False) as data:
        required = {
            "format_version",
            "f",
            "nside",
            "theta_min_deg",
            "log_relative_edges",
            "requested_repop_ids",
            "requested_scenarios",
            "input_root",
            "population_selection",
            "max_vmax_km_s",
            "catalogue_repop_ids",
            "catalogue_scenarios",
            "source_h5",
            "js_max_cat",
            "j_pixel_ref",
            "n_input_rows",
            "n_valid_rows",
            "n_physical_valid_rows",
            "n_vmax_excluded",
            "n_invalid_input_rows",
        }
        for population in POPULATIONS:
            required.add(f"hist_{population}")
            required.update(
                f"{field}_{population}"
                for field in POPULATION_SCALAR_FIELDS
            )
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(
                f"Checkpoint {path} is missing fields: {missing}"
            )

        compatibility = (
            int(data["format_version"]) == FORMAT_VERSION
            and float(data["f"]) == metadata["f"]
            and int(data["nside"]) == metadata["nside"]
            and float(data["theta_min_deg"]) == metadata["theta_min_deg"]
            and np.array_equal(
                data["log_relative_edges"],
                metadata["log_relative_edges"],
            )
            and np.array_equal(
                data["requested_repop_ids"],
                metadata["requested_repop_ids"],
            )
            and tuple(str(value) for value in data["requested_scenarios"])
            == metadata["requested_scenarios"]
            and str(data["input_root"]) == metadata["input_root"]
            and str(data["population_selection"])
            == metadata["population_selection"]
            and (
                (
                    np.isnan(float(data["max_vmax_km_s"]))
                    and np.isnan(metadata["max_vmax_km_s"])
                )
                or float(data["max_vmax_km_s"])
                == metadata["max_vmax_km_s"]
            )
        )
        if not compatibility:
            raise ValueError(
                f"Checkpoint {path} is incompatible with this scan."
            )

        repop_ids = np.asarray(data["catalogue_repop_ids"], dtype=np.int64)
        scenarios = np.asarray(data["catalogue_scenarios"], dtype=str)
        n_catalogues = len(repop_ids)
        if len(scenarios) != n_catalogues:
            raise ValueError(f"Inconsistent catalogue arrays in {path}.")

        for population in POPULATIONS:
            expected_shape = (
                n_catalogues,
                len(metadata["log_relative_edges"]) - 1,
            )
            if data[f"hist_{population}"].shape != expected_shape:
                raise ValueError(
                    f"Invalid hist_{population} shape in {path}: "
                    f"{data[f'hist_{population}'].shape}"
                )

        entries = []
        for index in range(n_catalogues):
            entry: dict[str, object] = {
                "repop_id": int(repop_ids[index]),
                "scenario": str(scenarios[index]),
                "source_h5": str(data["source_h5"][index]),
                "js_max_cat": float(data["js_max_cat"][index]),
                "j_pixel_ref": float(data["j_pixel_ref"][index]),
                "n_input_rows": int(data["n_input_rows"][index]),
                "n_valid_rows": int(data["n_valid_rows"][index]),
                "n_physical_valid_rows": int(
                    data["n_physical_valid_rows"][index]
                ),
                "n_vmax_excluded": int(data["n_vmax_excluded"][index]),
                "n_invalid_input_rows": int(
                    data["n_invalid_input_rows"][index]
                ),
            }
            for population in POPULATIONS:
                entry[f"hist_{population}"] = np.asarray(
                    data[f"hist_{population}"][index], dtype=np.int64
                )
                for field in POPULATION_SCALAR_FIELDS:
                    value = data[f"{field}_{population}"][index]
                    entry[f"{field}_{population}"] = (
                        int(value)
                        if field.startswith("n_")
                        else float(value)
                    )
            entries.append(entry)

    keys = [(entry["scenario"], entry["repop_id"]) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate catalogue entries in checkpoint {path}.")

    return entries


def run_catalogues(
    args: argparse.Namespace,
    repop_ids: np.ndarray,
    log_relative_edges: np.ndarray,
    process_catalogue_fn=None,
) -> list[dict[str, object]]:
    if process_catalogue_fn is None:
        process_catalogue_fn = process_catalogue

    metadata = checkpoint_metadata(args, repop_ids, log_relative_edges)
    entries = load_checkpoint(args.output_npz, metadata)
    completed = {
        (entry["scenario"], entry["repop_id"]): entry
        for entry in entries
    }
    requested = [
        (scenario, int(repop_id))
        for scenario in args.scenarios
        for repop_id in repop_ids
    ]

    for sequence, (scenario, repop_id) in enumerate(requested, start=1):
        key = (scenario, repop_id)
        source_h5 = catalogue_path(args.input_root, repop_id, scenario).resolve()

        if key in completed:
            entry = completed[key]
            if entry["source_h5"] != str(source_h5):
                raise ValueError(
                    f"Checkpoint source mismatch for {key}: "
                    f"{entry['source_h5']} != {source_h5}"
                )
            status = "checkpoint"
        else:
            if not source_h5.is_file():
                raise FileNotFoundError(f"Input HDF5 not found: {source_h5}")
            print(
                f"[{sequence:04d}/{len(requested):04d}] "
                f"repop_{repop_id:04d} {scenario}: scanning",
                flush=True,
            )
            result = process_catalogue_fn(
                path=source_h5,
                nside=args.nside,
                f_value=args.f,
                chunk_size=args.chunk_size,
                log_relative_edges=log_relative_edges,
                max_vmax_km_s=args.max_vmax_km_s,
            )
            entry = {
                "repop_id": repop_id,
                "scenario": scenario,
                "source_h5": str(source_h5),
                **result,
            }
            entries.append(entry)
            completed[key] = entry
            save_checkpoint(args.output_npz, metadata, entries)
            status = "saved"

        print(
            f"[{sequence:04d}/{len(requested):04d}] "
            f"repop_{repop_id:04d} {scenario}: {status} | "
            f"discarded={entry['n_discarded_all']:,}",
            flush=True,
        )

    return entries


def maximum_discarded_by_scenario(
    entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return exact maxima and catalogue IDs for both plotted populations."""
    summary = []
    scenarios = sorted({str(entry["scenario"]) for entry in entries})
    for scenario in scenarios:
        scenario_entries = [
            entry for entry in entries if entry["scenario"] == scenario
        ]
        for population in ("pointlike", "extended"):
            finite = [
                entry
                for entry in scenario_entries
                if np.isfinite(
                    float(entry[f"max_j_rel_discarded_{population}"])
                )
            ]
            if not finite:
                continue
            maximum = max(
                finite,
                key=lambda entry: float(
                    entry[f"max_j_rel_discarded_{population}"]
                ),
            )
            summary.append(
                {
                    "scenario": scenario,
                    "population": population,
                    "repop_id": int(maximum["repop_id"]),
                    "max_j_rel_discarded": float(
                        maximum[f"max_j_rel_discarded_{population}"]
                    ),
                    "max_js_discarded": float(
                        maximum[f"max_js_discarded_{population}"]
                    ),
                    "js_max_cat": float(maximum["js_max_cat"]),
                }
            )
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.input_root = args.input_root.resolve()
    args.output_npz = args.output_npz.resolve()

    repop_ids = np.arange(
        args.repop_start,
        args.repop_start + args.n_repops,
        dtype=np.int64,
    )
    log_relative_edges = np.linspace(
        args.log_rel_min,
        args.log_rel_max,
        args.n_bins + 1,
        dtype=np.float64,
    )

    entries = run_catalogues(
        args=args,
        repop_ids=repop_ids,
        log_relative_edges=log_relative_edges,
    )
    print(
        f"Saved {len(entries)} catalogue histogram(s): {args.output_npz}",
        flush=True,
    )
    print("Maximum discarded Js/Js,max by scenario:", flush=True)
    for row in maximum_discarded_by_scenario(entries):
        print(
            f"  {row['scenario']} {row['population']}: "
            f"{row['max_j_rel_discarded']:.8e} "
            f"(repop_{row['repop_id']:04d})",
            flush=True,
        )


if __name__ == "__main__":
    main()
