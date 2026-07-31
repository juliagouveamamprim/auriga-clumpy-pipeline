#!/usr/bin/env python3
"""Catalogue-wise central-pixel scan for pointlike and extended cuts."""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.prepare_subhalo_components import (  # noqa: E402
    CHUNK_SIZE,
    ITERATION,
    ROUND_UP_DECIMALS,
    build_valid_mask,
    compute_pixel_reference,
    get_input_h5,
    healpix_pixel_size_deg,
    clumpy_central_pixel_proxy_from_js,
    round_up,
    xyz_to_lb_deg,
)


DEFAULT_F_VALUES = "1e-5,1e-4,1e-3,1e-2"
VALID_EXTENDED_ENVELOPE_MODES = {"theta-s"}


def parse_f_values(text):
    values = sorted({float(value.strip()) for value in text.split(",")})

    if not values:
        raise ValueError("At least one cut fraction must be provided.")

    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Cut fractions must be finite and non-negative.")

    return values


def parse_envelope_modes(text):
    normalized = text.strip().lower()

    if normalized in {"", "none"}:
        return []

    modes = list(
        dict.fromkeys(
            value.strip()
            for value in normalized.split(",")
            if value.strip()
        )
    )

    invalid = sorted(
        set(modes) - VALID_EXTENDED_ENVELOPE_MODES
    )

    if invalid:
        raise ValueError(
            "Invalid extended envelope mode(s): "
            + ", ".join(invalid)
        )

    return modes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pointlike and extended J-factor cuts using a combined "
            "NSIDE HEALPix proxy map. Pointlike halos contribute Js to their "
            "pixel; extended halos contribute Jtheta to their central pixel."
        )
    )

    parser.add_argument("repop_id", type=int)
    parser.add_argument("scenario", choices=["resilient", "fragile"])

    parser.add_argument(
        "--input-h5",
        type=Path,
        default=None,
        help="Optional explicit HDF5 path.",
    )

    parser.add_argument(
        "--nside",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--pointlike-f-values",
        default=DEFAULT_F_VALUES,
        help="Comma-separated pointlike cut fractions.",
    )

    parser.add_argument(
        "--extended-f-values",
        default=DEFAULT_F_VALUES,
        help="Comma-separated extended cut fractions.",
    )

    parser.add_argument(
        "--theta-aperture-deg",
        type=float,
        default=None,
        help=(
            "Deprecated. The extended proxy now uses the CLUMPY "
            "aperture hp.max_pixrad(NSIDE), derived automatically."
        ),
    )

    parser.add_argument(
        "--extended-envelope-modes",
        default="none",
        help=(
            "Conservative envelope mode: 'theta-s' or 'none'."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def add_weighted_pixels(target, pixel, weights):
    if pixel.size == 0:
        return

    target += np.bincount(
        pixel,
        weights=weights,
        minlength=target.size,
    )


def add_extended_envelope_maps(
    target_maps,
    cuts,
    nside,
    lon_deg,
    lat_deg,
    theta_s_deg,
    weights,
):
    """Repeat discarded extended contributions across selected footprints."""
    if not target_maps or weights.size == 0:
        return

    candidate = weights < max(cuts.values())

    if not np.any(candidate):
        return

    candidate_lon = lon_deg[candidate]
    candidate_lat = lat_deg[candidate]
    candidate_theta_s = theta_s_deg[candidate]
    candidate_weights = weights[candidate]

    vectors = hp.ang2vec(
        candidate_lon,
        candidate_lat,
        lonlat=True,
    )

    for index, (vector, weight) in enumerate(
        zip(vectors, candidate_weights)
    ):
        for mode_maps in target_maps.values():
            radius_deg = candidate_theta_s[index]

            if not np.isfinite(radius_deg) or radius_deg <= 0.0:
                continue

            if radius_deg >= 180.0:
                pixels = None
            else:
                pixels = hp.query_disc(
                    nside,
                    vector,
                    np.deg2rad(radius_deg),
                    inclusive=True,
                    nest=True,
                )

            for f_value, j_cut in cuts.items():
                if weight >= j_cut:
                    continue

                if pixels is None:
                    mode_maps[f_value] += weight
                else:
                    mode_maps[f_value][pixels] += weight


def pixel_coordinates(nside, pixel):
    lon, lat = hp.pix2ang(
        nside,
        int(pixel),
        lonlat=True,
        nest=True,
    )
    return float(lon), float(lat)


def validate_nside(nside):
    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of two.")


def reduction_factor(total, kept):
    if kept == 0:
        return float("inf")
    return total / kept


def main():
    args = parse_args()

    if args.repop_id < 0:
        raise ValueError("repop_id must be non-negative.")

    validate_nside(args.nside)

    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive.")

    pointlike_f_values = parse_f_values(args.pointlike_f_values)
    extended_f_values = parse_f_values(args.extended_f_values)
    extended_envelope_modes = parse_envelope_modes(
        args.extended_envelope_modes
    )

    input_h5 = (
        args.input_h5.resolve()
        if args.input_h5 is not None
        else get_input_h5(args.repop_id, args.scenario)
    )

    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_h5}")

    theta_pix_deg = healpix_pixel_size_deg(args.nside)
    theta_min_deg = round_up(
        theta_pix_deg,
        decimals=ROUND_UP_DECIMALS,
    )

    if args.theta_aperture_deg is not None:
        raise ValueError(
            "--theta-aperture-deg is deprecated. The aperture is now "
            "derived automatically as hp.max_pixrad(NSIDE)."
        )

    alpha_int_rad = float(hp.max_pixrad(args.nside))
    alpha_int_deg = float(np.rad2deg(alpha_int_rad))

    # Retain the old CSV field name temporarily for compatibility
    # with the multi-repopulation aggregation script.
    theta_aperture_deg = alpha_int_deg

    npix = hp.nside2npix(args.nside)

    output_csv = args.output_csv

    if output_csv is None:
        output_csv = (
            REPOSITORY_ROOT
            / "diagnostics"
            / "cuts"
            / "results_summary"
            / (
                f"repop_{args.repop_id:04d}_{args.scenario}"
                f"_nside{args.nside}_central_pixel_scan.csv"
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    map_count_during_scan = (
        2
        + len(pointlike_f_values)
        + len(extended_f_values)
        + len(extended_envelope_modes) * len(extended_f_values)
    )

    map_count_during_evaluation = (
        len(pointlike_f_values)
        + len(extended_f_values)
        + len(extended_envelope_modes) * len(extended_f_values)
        + 2
        + int(bool(extended_envelope_modes))
    )

    peak_map_count = max(
        map_count_during_scan,
        map_count_during_evaluation,
    )
    estimated_gib = peak_map_count * npix * 8 / 1024**3

    print("=" * 80)
    print("Catalogue-wise combined cut scan")
    print("=" * 80)
    print(f"Input: {input_h5}")
    print(f"Scenario: {args.scenario}")
    print(f"Repopulation: {args.repop_id:04d}")
    print(f"NSIDE: {args.nside}")
    print(f"NPIX: {npix:,}")
    print(f"Pixel size sqrt(area): {theta_pix_deg:.8f} deg")
    print(f"Pointlike/extended threshold: {theta_min_deg:.8f} deg")
    print(
        "CLUMPY integration aperture: "
        f"{theta_aperture_deg:.12f} deg"
    )
    print(f"Pointlike f values: {pointlike_f_values}")
    print(f"Extended f values: {extended_f_values}")
    print(
        "Extended envelope modes: "
        + (
            ", ".join(extended_envelope_modes)
            if extended_envelope_modes
            else "none"
        )
    )
    print(f"Chunk size: {args.chunk_size:,}")
    print(
        "Estimated peak persistent map memory: "
        f"{estimated_gib:.2f} GiB"
    )
    print("=" * 80)

    group_name = f"iteration_{ITERATION}"

    with h5py.File(input_h5, "r") as h5:
        if group_name not in h5:
            raise KeyError(f"Missing group: {group_name}")

        group = h5[group_name]

        if "data" not in group:
            raise KeyError(f"Missing dataset: {group_name}/data")

        data = group["data"]

        if data.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional data table, got {data.shape}."
            )

        if "column_names" not in data.attrs:
            raise KeyError("The data dataset has no column_names attribute.")

        column_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in data.attrs["column_names"]
        ]

        required_columns = [
            "Js",
            "D_Earth",
            "theta_s",
            "r_s",
            "rho_s",
            "Xearth",
            "Yearth",
            "Zearth",
        ]

        missing_columns = [
            name for name in required_columns
            if name not in column_names
        ]

        if missing_columns:
            raise KeyError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        column_indices = {
            name: column_names.index(name)
            for name in required_columns
        }

        n_total = data.shape[0]

        print(
            "[1/3] Computing J_pixel_ref",
            flush=True,
        )

        ref_info = compute_pixel_reference(
            data=data,
            column_indices=column_indices,
            n_total=n_total,
            theta_min_deg=theta_min_deg,
            nside=args.nside,
            chunk_size=args.chunk_size,
            progress_label="Reference scan",
        )

        j_pixel_ref = ref_info["j_pixel_ref"]

        print(
            "Brightest pointlike proxy: "
            f"{ref_info['brightest_pointlike']:.8e}"
        )
        print(
            "Brightest extended proxy:  "
            f"{ref_info['brightest_extended']:.8e}"
        )
        print(f"J_pixel_ref:               {j_pixel_ref:.8e}")

        pointlike_cuts = {
            f_value: f_value * j_pixel_ref
            for f_value in pointlike_f_values
        }
        extended_cuts = {
            f_value: f_value * j_pixel_ref
            for f_value in extended_f_values
        }

        pointlike_full = np.zeros(npix, dtype=np.float64)
        extended_full = np.zeros(npix, dtype=np.float64)

        pointlike_discarded_maps = {
            f_value: np.zeros(npix, dtype=np.float64)
            for f_value in pointlike_f_values
        }
        extended_discarded_maps = {
            f_value: np.zeros(npix, dtype=np.float64)
            for f_value in extended_f_values
        }

        extended_envelope_maps = {
            mode: {
                f_value: np.zeros(npix, dtype=np.float64)
                for f_value in extended_f_values
            }
            for mode in extended_envelope_modes
        }

        n_valid_total = 0
        n_pointlike_total = 0
        n_extended_total = 0

        pointlike_discarded_counts = {
            f_value: 0
            for f_value in pointlike_f_values
        }
        extended_discarded_counts = {
            f_value: 0
            for f_value in extended_f_values
        }

        pointlike_discarded_sums = {
            f_value: 0.0
            for f_value in pointlike_f_values
        }
        extended_discarded_sums = {
            f_value: 0.0
            for f_value in extended_f_values
        }

        print(
            "[2/3] Building central-pixel and envelope maps",
            flush=True,
        )

        for start_row in range(0, n_total, args.chunk_size):
            end_row = min(start_row + args.chunk_size, n_total)
            arr = data[start_row:end_row]

            js = arr[:, column_indices["Js"]]
            d_earth = arr[:, column_indices["D_Earth"]]
            theta_s = arr[:, column_indices["theta_s"]]
            r_s = arr[:, column_indices["r_s"]]
            rho_s = arr[:, column_indices["rho_s"]]
            x_e = arr[:, column_indices["Xearth"]]
            y_e = arr[:, column_indices["Yearth"]]
            z_e = arr[:, column_indices["Zearth"]]

            valid = build_valid_mask(
                js=js,
                d_earth=d_earth,
                theta_s=theta_s,
                r_s=r_s,
                rho_s=rho_s,
                x_e=x_e,
                y_e=y_e,
                z_e=z_e,
            )

            mask_pointlike = valid & (theta_s < theta_min_deg)
            mask_extended = valid & (theta_s >= theta_min_deg)

            n_valid_total += int(np.count_nonzero(valid))
            n_pointlike_total += int(np.count_nonzero(mask_pointlike))
            n_extended_total += int(np.count_nonzero(mask_extended))

            if np.any(mask_pointlike):
                pointlike_js = js[mask_pointlike]

                pointlike_lon, pointlike_lat = xyz_to_lb_deg(
                    x_e[mask_pointlike],
                    y_e[mask_pointlike],
                    z_e[mask_pointlike],
                )

                pointlike_pixel = hp.ang2pix(
                    args.nside,
                    pointlike_lon,
                    pointlike_lat,
                    lonlat=True,
                    nest=True,
                )

                add_weighted_pixels(
                    pointlike_full,
                    pointlike_pixel,
                    pointlike_js,
                )

                for f_value, j_cut in pointlike_cuts.items():
                    discarded = pointlike_js < j_cut

                    if not np.any(discarded):
                        continue

                    add_weighted_pixels(
                        pointlike_discarded_maps[f_value],
                        pointlike_pixel[discarded],
                        pointlike_js[discarded],
                    )

                    pointlike_discarded_counts[f_value] += int(
                        np.count_nonzero(discarded)
                    )
                    pointlike_discarded_sums[f_value] += float(
                        pointlike_js[discarded].sum(dtype=np.float64)
                    )

            if np.any(mask_extended):
                extended_js = js[mask_extended]

                extended_pixel_proxy = (
                    clumpy_central_pixel_proxy_from_js(
                        js=extended_js,
                        d_earth_kpc=d_earth[mask_extended],
                        rs_kpc=r_s[mask_extended],
                        nside=args.nside,
                    )
                )

                extended_lon, extended_lat = xyz_to_lb_deg(
                    x_e[mask_extended],
                    y_e[mask_extended],
                    z_e[mask_extended],
                )

                extended_pixel = hp.ang2pix(
                    args.nside,
                    extended_lon,
                    extended_lat,
                    lonlat=True,
                    nest=True,
                )

                add_weighted_pixels(
                    extended_full,
                    extended_pixel,
                    extended_pixel_proxy,
                )

                add_extended_envelope_maps(
                    target_maps=extended_envelope_maps,
                    cuts=extended_cuts,
                    nside=args.nside,
                    lon_deg=extended_lon,
                    lat_deg=extended_lat,
                    theta_s_deg=theta_s[mask_extended],
                    weights=extended_pixel_proxy,
                )

                for f_value, j_cut in extended_cuts.items():
                    discarded = extended_pixel_proxy < j_cut

                    if not np.any(discarded):
                        continue

                    add_weighted_pixels(
                        extended_discarded_maps[f_value],
                        extended_pixel[discarded],
                        extended_pixel_proxy[discarded],
                    )

                    extended_discarded_counts[f_value] += int(
                        np.count_nonzero(discarded)
                    )
                    extended_discarded_sums[f_value] += float(
                        extended_pixel_proxy[discarded].sum(dtype=np.float64)
                    )

            print(
                f"Rows {start_row:,}-{end_row:,}/{n_total:,} | "
                f"pointlike={n_pointlike_total:,} | "
                f"extended={n_extended_total:,}",
                flush=True,
            )

    full_combined = pointlike_full + extended_full

    full_sum = float(full_combined.sum(dtype=np.float64))
    full_max = float(full_combined.max())

    pointlike_full_sum = float(pointlike_full.sum(dtype=np.float64))
    extended_full_sum = float(extended_full.sum(dtype=np.float64))

    del pointlike_full
    del extended_full

    temporary_map = np.empty(npix, dtype=np.float64)
    envelope_temporary_map = (
        np.empty(npix, dtype=np.float64)
        if extended_envelope_modes
        else None
    )
    rows = []

    print()
    print(
        "[3/3] Evaluating combined cut pairs",
        flush=True,
    )

    for pointlike_f in pointlike_f_values:
        pointlike_discarded = pointlike_discarded_maps[pointlike_f]

        n_pointlike_discarded = pointlike_discarded_counts[pointlike_f]
        n_pointlike_kept = (
            n_pointlike_total - n_pointlike_discarded
        )

        for extended_f in extended_f_values:
            extended_discarded = extended_discarded_maps[extended_f]

            n_extended_discarded = extended_discarded_counts[extended_f]
            n_extended_kept = (
                n_extended_total - n_extended_discarded
            )

            np.add(
                pointlike_discarded,
                extended_discarded,
                out=temporary_map,
            )

            pixel_max_discarded = int(np.argmax(temporary_map))
            max_discarded_total = float(
                temporary_map[pixel_max_discarded]
            )
            discarded_sum = float(
                temporary_map.sum(dtype=np.float64)
            )

            discarded_lon, discarded_lat = pixel_coordinates(
                args.nside,
                pixel_max_discarded,
            )

            np.subtract(
                full_combined,
                temporary_map,
                out=temporary_map,
            )

            pixel_max_final = int(np.argmax(temporary_map))
            max_final = float(temporary_map[pixel_max_final])

            final_lon, final_lat = pixel_coordinates(
                args.nside,
                pixel_max_final,
            )

            ratio_max_discarded_to_final = (
                max_discarded_total / max_final
                if max_final > 0.0
                else float("inf")
            )

            envelope_results = {}

            for mode in extended_envelope_modes:
                mode_key = mode.replace("-", "_")
                extended_envelope = (
                    extended_envelope_maps[mode][extended_f]
                )

                np.add(
                    pointlike_discarded,
                    extended_envelope,
                    out=envelope_temporary_map,
                )

                envelope_pixel = int(
                    np.argmax(envelope_temporary_map)
                )
                envelope_max = float(
                    envelope_temporary_map[envelope_pixel]
                )
                envelope_lon, envelope_lat = pixel_coordinates(
                    args.nside,
                    envelope_pixel,
                )

                envelope_ratio = (
                    envelope_max / max_final
                    if max_final > 0.0
                    else float("inf")
                )

                envelope_results.update(
                    {
                        (
                            "max_discarded_extended_"
                            f"{mode_key}_envelope_pixel"
                        ): float(extended_envelope.max()),
                        (
                            "max_discarded_combined_"
                            f"{mode_key}_envelope_pixel"
                        ): envelope_max,
                        (
                            "pixel_max_discarded_combined_"
                            f"{mode_key}_envelope"
                        ): envelope_pixel,
                        (
                            "lon_max_discarded_combined_"
                            f"{mode_key}_envelope_deg"
                        ): envelope_lon,
                        (
                            "lat_max_discarded_combined_"
                            f"{mode_key}_envelope_deg"
                        ): envelope_lat,
                        (
                            "ratio_max_discarded_"
                            f"{mode_key}_envelope_to_final"
                        ): envelope_ratio,
                    }
                )

            fraction_sum_discarded_to_full = (
                discarded_sum / full_sum
                if full_sum > 0.0
                else float("nan")
            )

            row = {
                "scenario": args.scenario,
                "repop_id": args.repop_id,
                "input_h5": str(input_h5),
                "nside": args.nside,
                "npix": npix,
                "theta_pix_deg": theta_pix_deg,
                "theta_min_deg": theta_min_deg,
                "theta_aperture_deg": theta_aperture_deg,
                "extended_envelope_modes": (
                    ",".join(extended_envelope_modes)
                    if extended_envelope_modes
                    else "none"
                ),
                "j_pixel_ref": j_pixel_ref,
                "brightest_pointlike_proxy": (
                    ref_info["brightest_pointlike"]
                ),
                "brightest_extended_proxy": (
                    ref_info["brightest_extended"]
                ),
                "pointlike_f": pointlike_f,
                "extended_f": extended_f,
                "pointlike_j_cut": pointlike_cuts[pointlike_f],
                "extended_j_cut": extended_cuts[extended_f],
                "n_valid_total": n_valid_total,
                "n_pointlike_total": n_pointlike_total,
                "n_pointlike_kept": n_pointlike_kept,
                "n_pointlike_discarded": n_pointlike_discarded,
                "pointlike_reduction_factor": reduction_factor(
                    n_pointlike_total,
                    n_pointlike_kept,
                ),
                "n_extended_total": n_extended_total,
                "n_extended_kept": n_extended_kept,
                "n_extended_discarded": n_extended_discarded,
                "extended_reduction_factor": reduction_factor(
                    n_extended_total,
                    n_extended_kept,
                ),
                "pointlike_full_sum": pointlike_full_sum,
                "extended_central_full_sum": extended_full_sum,
                "full_combined_sum": full_sum,
                "full_combined_max": full_max,
                "discarded_pointlike_sum": (
                    pointlike_discarded_sums[pointlike_f]
                ),
                "discarded_extended_central_sum": (
                    extended_discarded_sums[extended_f]
                ),
                "discarded_combined_sum": discarded_sum,
                "fraction_sum_discarded_to_full": (
                    fraction_sum_discarded_to_full
                ),
                "max_discarded_pointlike_pixel": float(
                    pointlike_discarded.max()
                ),
                "max_discarded_extended_central_pixel": float(
                    extended_discarded.max()
                ),
                "max_discarded_combined_pixel": max_discarded_total,
                "pixel_max_discarded_combined": pixel_max_discarded,
                "lon_max_discarded_combined_deg": discarded_lon,
                "lat_max_discarded_combined_deg": discarded_lat,
                "max_final_combined_pixel": max_final,
                "pixel_max_final_combined": pixel_max_final,
                "lon_max_final_combined_deg": final_lon,
                "lat_max_final_combined_deg": final_lat,
                "ratio_max_discarded_to_final": (
                    ratio_max_discarded_to_final
                ),
            }

            row.update(envelope_results)
            rows.append(row)

            envelope_summary_parts = []

            for mode in extended_envelope_modes:
                mode_key = mode.replace("-", "_")
                ratio_key = (
                    "ratio_max_discarded_"
                    f"{mode_key}_envelope_to_final"
                )
                envelope_summary_parts.append(
                    f" | {mode}="
                    f"{envelope_results[ratio_key]:.6e}"
                )

            envelope_summary = "".join(
                envelope_summary_parts
            )

            print(
                f"f_pl={pointlike_f:.1e} "
                f"f_ext={extended_f:.1e} | "
                f"N_pl={n_pointlike_kept:,} "
                f"N_ext={n_extended_kept:,} | "
                f"central={ratio_max_discarded_to_final:.6e}"
                f"{envelope_summary}"
            )

    fieldnames = list(rows[0].keys())

    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("Finished combined cut scan")
    print("=" * 80)
    print(f"Output CSV: {output_csv}")
    print(f"Rows written: {len(rows)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
