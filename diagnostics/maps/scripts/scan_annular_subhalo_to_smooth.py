#!/usr/bin/env python3

"""Measure annular rendered-subhalo-to-smooth ratios in HEALPix maps."""

import argparse
import csv
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.io import fits


HDU_NAME = "JFACTOR"
PIXEL_COLUMN = "PIXEL"
COMPONENT_COLUMNS = {
    "smooth": "Jsmooth",
    "pointlike": "Jpointlike",
    "extended": "Jextended",
}
SCENARIOS = ("fragile", "resilient")
CSV_COLUMNS = (
    "scenario",
    "repop_id",
    "psi_min_deg",
    "psi_max_deg",
    "psi_center_deg",
    "n_pixels",
    "sum_j_smooth",
    "sum_j_pointlike",
    "sum_j_extended",
    "sum_j_sub",
    "ratio_sub_to_smooth",
)


@dataclass(frozen=True)
class MapView:
    """Memory-mapped component columns from one final pipeline FITS."""

    path: Path
    values: dict
    pixels: object
    nside: int
    npix: int
    ordering: str
    units: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan smooth, pointlike, and extended HEALPix maps for one "
            "fragile/resilient repopulation and save annular rendered "
            "J_sub/J_smooth."
        )
    )
    parser.add_argument("--repop-id", type=int, required=True)
    for scenario in SCENARIOS:
        parser.add_argument(
            f"--{scenario}-map",
            type=Path,
            required=True,
            help=(
                f"Final pipeline FITS containing Jsmooth, Jpointlike, and "
                f"Jextended in the {HDU_NAME} HDU for the {scenario} "
                "scenario."
            ),
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which the diagnostic CSV will be written.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "CSV filename (default: annular_subhalo_to_smooth_"
            "repop_XXXX.csv)."
        ),
    )
    parser.add_argument(
        "--bin-width-deg",
        type=float,
        default=1.0,
        help="Nominal angular width in degrees (default: 1).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of HEALPix rows processed at once (default: 1000000).",
    )
    return parser.parse_args()


def normalize_ordering(value, label):
    ordering = str(value or "").strip().upper()
    if ordering in {"NEST", "NESTED"}:
        return "NEST"
    if ordering == "RING":
        return "RING"
    raise ValueError(
        f"{label}: ORDERING must be RING or NESTED, got {value!r}."
    )


def parse_unit(value, label):
    if value is None or not str(value).strip():
        raise ValueError(f"{label}: the J-factor column has no unit.")
    try:
        return u.Unit(value, format="fits")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label}: invalid FITS unit {value!r}."
        ) from error


def open_map_view(hdul, path):
    label = f"{path} [{HDU_NAME}]"

    if HDU_NAME not in hdul:
        raise ValueError(f"{path}: missing {HDU_NAME!r} HDU.")
    hdu = hdul[HDU_NAME]
    if hdu.data is None:
        raise ValueError(f"{path}: {HDU_NAME!r} HDU has no table data.")
    if PIXEL_COLUMN not in hdu.columns.names:
        raise ValueError(
            f"{path}: {HDU_NAME!r} HDU is missing {PIXEL_COLUMN!r}."
        )
    for column in COMPONENT_COLUMNS.values():
        if column not in hdu.columns.names:
            raise ValueError(
                f"{path}: {HDU_NAME!r} HDU is missing {column!r}."
            )

    if str(hdu.header.get("PIXTYPE", "")).strip().upper() != "HEALPIX":
        raise ValueError(f"{label}: PIXTYPE is not HEALPIX.")
    if str(hdu.header.get("INDXSCHM", "")).strip().upper() != "EXPLICIT":
        raise ValueError(f"{label}: INDXSCHM must be EXPLICIT.")
    if str(hdu.header.get("COORDSYS", "")).strip().upper() != "G":
        raise ValueError(f"{label}: COORDSYS must be Galactic (G).")

    try:
        nside = int(hdu.header["NSIDE"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: missing or invalid NSIDE.") from error
    if not hp.isnsideok(nside):
        raise ValueError(f"{label}: invalid HEALPix NSIDE={nside}.")

    npix = len(hdu.data)
    expected_npix = hp.nside2npix(nside)
    if npix != expected_npix:
        raise ValueError(
            f"{label}: found {npix} rows, expected {expected_npix} "
            f"for NSIDE={nside}."
        )

    return MapView(
        path=path,
        values={
            component: hdu.data[column]
            for component, column in COMPONENT_COLUMNS.items()
        },
        pixels=hdu.data[PIXEL_COLUMN],
        nside=nside,
        npix=npix,
        ordering=normalize_ordering(hdu.header.get("ORDERING"), label),
        units={
            component: parse_unit(
                hdu.columns[column].unit,
                f"{label}:{column}",
            )
            for component, column in COMPONENT_COLUMNS.items()
        },
    )


def unit_conversion_factor(source, target, label):
    try:
        return float(source.to(target))
    except u.UnitConversionError as error:
        raise ValueError(
            f"{label}: incompatible units {source!s} and {target!s}."
        ) from error


def validate_component_units(scenario, view):
    reference_unit = view.units["smooth"]
    conversion_factors = {}

    for component, unit in view.units.items():
        label = f"{scenario} {component}"
        conversion_factors[component] = unit_conversion_factor(
            unit,
            reference_unit,
            label,
        )

    return conversion_factors


def validate_scenario_compatibility(views):
    reference = views["fragile"]
    other = views["resilient"]
    if other.nside != reference.nside:
        raise ValueError(
            f"resilient NSIDE={other.nside} differs from fragile "
            f"NSIDE={reference.nside}."
        )
    if other.npix != reference.npix:
        raise ValueError(
            f"resilient number of pixels {other.npix} differs from fragile "
            f"number of pixels {reference.npix}."
        )
    if other.ordering != reference.ordering:
        raise ValueError(
            f"resilient ORDERING={other.ordering} differs from fragile "
            f"ORDERING={reference.ordering}."
        )


def angular_bin_edges(bin_width_deg):
    if not np.isfinite(bin_width_deg) or not 0.0 < bin_width_deg <= 180.0:
        raise ValueError("bin-width-deg must be finite and in (0, 180].")

    edges = np.arange(0.0, 180.0, bin_width_deg, dtype=np.float64)
    if edges.size == 0 or edges[0] != 0.0:
        edges = np.insert(edges, 0, 0.0)
    if not np.isclose(edges[-1], 180.0, rtol=0.0, atol=1e-12):
        edges = np.append(edges, 180.0)
    else:
        edges[-1] = 180.0
    return edges


def checked_values(view, component, start, stop, conversion_factor, scenario):
    values = np.asarray(view.values[component][start:stop], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{scenario} {component}: non-finite J-factor value "
            f"in rows {start}:{stop}."
        )
    if np.any(values < 0.0):
        raise ValueError(
            f"{scenario} {component}: negative J-factor value "
            f"in rows {start}:{stop}."
        )
    if conversion_factor != 1.0:
        values *= conversion_factor
    return values


def scan_scenario(scenario, view, repop_id, edges, chunk_size):
    """Return one output record per angular annulus for one scenario."""

    if chunk_size <= 0:
        raise ValueError("chunk-size must be a positive integer.")

    factors = validate_component_units(scenario, view)
    n_bins = len(edges) - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    sums = {
        component: np.zeros(n_bins, dtype=np.float64)
        for component in COMPONENT_COLUMNS
    }
    sum_sub_direct = np.zeros(n_bins, dtype=np.float64)
    nest = view.ordering == "NEST"

    for start in range(0, view.npix, chunk_size):
        stop = min(start + chunk_size, view.npix)
        pixels = np.asarray(view.pixels[start:stop], dtype=np.int64)
        expected_pixels = np.arange(start, stop, dtype=np.int64)
        if not np.array_equal(pixels, expected_pixels):
            raise ValueError(
                f"{scenario}: PIXEL must be the complete ordered "
                f"sequence 0..{view.npix - 1}; mismatch in rows "
                f"{start}:{stop}."
            )

        # At Galactic coordinates (l, b)=(0, 0), n_GC=(1, 0, 0), so
        # n_pixel dot n_GC is the Cartesian x component from pix2vec.
        cos_psi = hp.pix2vec(view.nside, pixels, nest=nest)[0]
        psi_deg = np.degrees(
            np.arccos(np.clip(cos_psi, -1.0, 1.0))
        )
        bin_index = np.searchsorted(edges, psi_deg, side="right") - 1
        np.clip(bin_index, 0, n_bins - 1, out=bin_index)
        counts += np.bincount(bin_index, minlength=n_bins)

        chunk_values = {}
        for component in COMPONENT_COLUMNS:
            values = checked_values(
                view,
                component,
                start,
                stop,
                factors[component],
                scenario,
            )
            chunk_values[component] = values
            sums[component] += np.bincount(
                bin_index,
                weights=values,
                minlength=n_bins,
            )

        sum_sub_direct += np.bincount(
            bin_index,
            weights=(
                chunk_values["pointlike"] + chunk_values["extended"]
            ),
            minlength=n_bins,
        )

    if int(counts.sum()) != view.npix:
        raise RuntimeError(
            f"{scenario}: annular pixel counts do not cover the full sky."
        )

    sum_sub_components = sums["pointlike"] + sums["extended"]
    if not np.allclose(
        sum_sub_direct,
        sum_sub_components,
        rtol=1e-10,
        atol=0.0,
    ):
        raise RuntimeError(
            f"{scenario}: annular sum_j_sub is inconsistent with "
            "sum_j_pointlike + sum_j_extended."
        )

    denominators = sums["smooth"]
    invalid = (~np.isfinite(denominators)) | (denominators <= 0.0)
    if np.any(invalid):
        bad_bins = np.flatnonzero(invalid)
        preview = ", ".join(str(int(index)) for index in bad_bins[:8])
        raise ValueError(
            f"{scenario}: non-positive or non-finite annular smooth "
            f"denominator in bin(s) {preview}."
        )

    ratios = sum_sub_direct / denominators
    if not np.all(np.isfinite(ratios)):
        raise ValueError(f"{scenario}: non-finite annular ratio.")

    records = []
    for index in range(n_bins):
        records.append(
            {
                "scenario": scenario,
                "repop_id": repop_id,
                "psi_min_deg": edges[index],
                "psi_max_deg": edges[index + 1],
                "psi_center_deg": 0.5 * (edges[index] + edges[index + 1]),
                "n_pixels": int(counts[index]),
                "sum_j_smooth": denominators[index],
                "sum_j_pointlike": sums["pointlike"][index],
                "sum_j_extended": sums["extended"][index],
                "sum_j_sub": sum_sub_direct[index],
                "ratio_sub_to_smooth": ratios[index],
            }
        )
    return records


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        format(value, ".17g")
                        if isinstance(value, (float, np.floating))
                        else value
                    )
                    for key, value in record.items()
                }
            )


def scan_all(map_paths, repop_id, bin_width_deg, chunk_size):
    edges = angular_bin_edges(bin_width_deg)
    records = []

    unique_paths = {
        Path(path).expanduser().resolve()
        for path in map_paths.values()
    }
    missing = [path for path in sorted(unique_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input FITS not found: {missing[0]}")

    repop_pattern = re.compile(r"repop_(\d+)")
    for path in unique_paths:
        path_ids = {
            int(match.group(1))
            for match in repop_pattern.finditer(str(path))
        }
        if path_ids and path_ids != {repop_id}:
            raise ValueError(
                f"Input path {path} identifies repopulation(s) "
                f"{sorted(path_ids)}, not requested repop_id={repop_id}."
            )

    with ExitStack() as stack:
        open_files = {
            path: stack.enter_context(fits.open(path, memmap=True))
            for path in unique_paths
        }
        views = {
            scenario: open_map_view(
                open_files[Path(map_paths[scenario]).expanduser().resolve()],
                Path(map_paths[scenario]).expanduser().resolve(),
            )
            for scenario in SCENARIOS
        }
        validate_scenario_compatibility(views)
        for scenario in SCENARIOS:
            records.extend(
                scan_scenario(
                    scenario,
                    views[scenario],
                    repop_id,
                    edges,
                    chunk_size,
                )
            )
    return records


def main():
    args = parse_args()
    if args.repop_id < 0:
        raise ValueError("repop-id must be a non-negative integer.")

    map_paths = {
        scenario: getattr(args, f"{scenario}_map")
        for scenario in SCENARIOS
    }
    records = scan_all(
        map_paths=map_paths,
        repop_id=args.repop_id,
        bin_width_deg=args.bin_width_deg,
        chunk_size=args.chunk_size,
    )
    output_name = args.output_name or (
        f"annular_subhalo_to_smooth_repop_{args.repop_id:04d}.csv"
    )
    if Path(output_name).name != output_name:
        raise ValueError("output-name must be a filename, not a path.")
    output_path = args.output_dir / output_name
    write_records(output_path, records)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
