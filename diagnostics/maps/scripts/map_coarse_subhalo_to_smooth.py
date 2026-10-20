#!/usr/bin/env python3

"""Map coarse all-sky rendered-subhalo-to-smooth J-factor ratios."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm


SCENARIOS = ("fragile", "resilient")
DEFAULT_INPUT_NSIDE = 2048
DEFAULT_DIAG_NSIDE = 16
DEFAULT_CHUNK_SIZE = 1_000_000
HDU_NAME = "JFACTOR"
PIXEL_COLUMN = "PIXEL"
SMOOTH_COLUMN = "Jsmooth"
POINTLIKE_COLUMN = "Jpointlike"
EXTENDED_COLUMN = "Jextended"
COMPONENT_COLUMNS = (SMOOTH_COLUMN, POINTLIKE_COLUMN, EXTENDED_COLUMN)

# CLUMPY's conceptual 1e-40 empty-Jlist marker is stored in the production
# float32 table as this exact value.  This is an equality match, not a floor.
CLUMPY_EMPTY_JLIST = np.float32(1.0e-40)

SUM_RTOL = 1.0e-12
PLOT_VMIN = -3.0
PLOT_VCENTER = 0.0
PLOT_VMAX = 1.5
COLORBAR_TICKS = (-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5)
SUMMARY_COLUMNS = (
    "scenario",
    "repop_id",
    "diagnostic_nside",
    "n_bins",
    "n_zero_substructure_bins",
    "min_finite_R",
    "max_finite_R",
    "n_bins_R_gt_1",
    "fraction_bins_R_gt_1",
)


@dataclass(frozen=True)
class MapMetadata:
    """Validated metadata needed to stream one final pipeline map."""

    nside: int
    npix: int
    children_per_parent: int
    coarse_npix: int
    conversion_factors: dict[str, float]


@dataclass(frozen=True)
class CoarseRatioMap:
    """Sum-conserving coarse components and their ratio for one scenario."""

    scenario: str
    repop_id: int
    input_nside: int
    diagnostic_nside: int
    j_smooth: np.ndarray
    j_sub: np.ndarray
    ratio: np.ndarray
    n_extended_sentinel_pixels: int


@dataclass(frozen=True)
class RatioSummary:
    scenario: str
    repop_id: int
    diagnostic_nside: int
    n_bins: int
    n_zero_substructure_bins: int
    min_finite_R: float
    max_finite_R: float
    n_bins_R_gt_1: int
    fraction_bins_R_gt_1: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate final integrated-J HEALPix maps into coarse NESTED "
            "bins and plot log10(J_sub/J_smooth) for fragile and resilient."
        )
    )
    parser.add_argument("--repop-id", type=int, required=True)
    parser.add_argument(
        "--maps-root",
        type=Path,
        default=Path("outputs/clumpy"),
        help="Root containing <scenario>/outputs/total (default: outputs/clumpy).",
    )
    parser.add_argument("--fragile-map", type=Path, default=None)
    parser.add_argument("--resilient-map", type=Path, default=None)
    parser.add_argument("--input-nside", type=int, default=DEFAULT_INPUT_NSIDE)
    parser.add_argument("--diag-nside", type=int, default=DEFAULT_DIAG_NSIDE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: outputs/diagnostics/maps/"
            "coarse_subhalo_to_smooth/repop_XXXX)."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    parser.add_argument("--basename", default=None)
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.repop_id < 0:
        raise ValueError("--repop-id must be a non-negative integer.")
    for label, nside in (
        ("input", args.input_nside),
        ("diagnostic", args.diag_nside),
    ):
        if not hp.isnsideok(nside, nest=True):
            raise ValueError(f"Invalid {label} NSIDE={nside}.")
    if args.input_nside < args.diag_nside:
        raise ValueError("Input NSIDE must be at least the diagnostic NSIDE.")
    if args.input_nside % args.diag_nside != 0:
        raise ValueError("Input NSIDE must be divisible by diagnostic NSIDE.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")
    if args.basename is not None and Path(args.basename).name != args.basename:
        raise ValueError("--basename must be a filename stem, not a path.")


def expected_map_path(
    maps_root: Path,
    scenario: str,
    repop_id: int,
    input_nside: int,
) -> Path:
    return (
        Path(maps_root)
        / scenario
        / "outputs"
        / "total"
        / f"repop_{repop_id:04d}_nside{input_nside}"
        / f"auriga_total_nside{input_nside}.fits"
    )


def default_output_dir(repop_id: int) -> Path:
    return (
        Path("outputs")
        / "diagnostics"
        / "maps"
        / "coarse_subhalo_to_smooth"
        / f"repop_{repop_id:04d}"
    )


def resolve_map_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        scenario: Path(getattr(args, f"{scenario}_map"))
        if getattr(args, f"{scenario}_map") is not None
        else expected_map_path(
            args.maps_root,
            scenario,
            args.repop_id,
            args.input_nside,
        )
        for scenario in SCENARIOS
    }


def normalize_ordering(value: object, label: str) -> str:
    ordering = str(value or "").strip().upper()
    if ordering in {"NEST", "NESTED"}:
        return "NEST"
    if ordering == "RING":
        return "RING"
    raise ValueError(f"{label}: ORDERING must be RING or NESTED, got {value!r}.")


def parse_unit(value: object, label: str) -> u.UnitBase:
    if value is None or not str(value).strip():
        raise ValueError(f"{label}: J-factor column has no unit.")
    try:
        return u.Unit(value, format="fits")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid FITS unit {value!r}.") from error


def integer_header_value(header: fits.Header, keyword: str, label: str) -> int:
    try:
        return int(header[keyword])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: missing or invalid {keyword}.") from error


def validate_column_format(
    column: fits.Column,
    expected_code: str,
    label: str,
) -> None:
    column_format = column.format
    if column_format.repeat != 1 or column_format.format != expected_code:
        raise ValueError(
            f"{label}: expected scalar FITS format 1{expected_code}, "
            f"found {column.format!s}."
        )
    if column.dim is not None:
        raise ValueError(f"{label}: unexpected TDIM={column.dim!r}.")
    if column.bscale not in (None, 1, 1.0):
        raise ValueError(f"{label}: unexpected TSCAL={column.bscale!r}.")
    if column.bzero not in (None, 0, 0.0):
        raise ValueError(f"{label}: unexpected TZERO={column.bzero!r}.")


def validate_map_hdu_metadata(
    hdu: fits.BinTableHDU,
    path: Path,
    input_nside: int,
    diagnostic_nside: int,
) -> MapMetadata:
    """Validate headers and column definitions without reading table data."""
    label = f"{path} [{HDU_NAME}]"
    if not isinstance(hdu, fits.BinTableHDU):
        raise ValueError(f"{label}: expected a BINTABLE HDU.")
    required = (PIXEL_COLUMN, *COMPONENT_COLUMNS)
    missing = [name for name in required if name not in hdu.columns.names]
    if missing:
        raise ValueError(f"{label}: missing column(s) {missing}.")
    if str(hdu.header.get("PIXTYPE", "")).strip().upper() != "HEALPIX":
        raise ValueError(f"{label}: PIXTYPE is not HEALPIX.")
    if str(hdu.header.get("INDXSCHM", "")).strip().upper() != "EXPLICIT":
        raise ValueError(f"{label}: INDXSCHM must be EXPLICIT.")
    if str(hdu.header.get("COORDSYS", "")).strip().upper() != "G":
        raise ValueError(f"{label}: COORDSYS must be Galactic (G).")
    if normalize_ordering(hdu.header.get("ORDERING"), label) != "NEST":
        raise ValueError(f"{label}: expected ORDERING=NESTED.")

    nside = integer_header_value(hdu.header, "NSIDE", label)
    if nside != input_nside:
        raise ValueError(f"{label}: expected NSIDE={input_nside}, found {nside}.")
    if not hp.isnsideok(nside, nest=True):
        raise ValueError(f"{label}: invalid HEALPix NSIDE={nside}.")
    expected_npix = hp.nside2npix(nside)
    n_rows = integer_header_value(hdu.header, "NAXIS2", label)
    if n_rows != expected_npix:
        raise ValueError(f"{label}: found {n_rows} rows, expected {expected_npix}.")
    if integer_header_value(hdu.header, "PCOUNT", label) != 0:
        raise ValueError(f"{label}: expected PCOUNT=0.")
    if integer_header_value(hdu.header, "GCOUNT", label) != 1:
        raise ValueError(f"{label}: expected GCOUNT=1.")
    row_width = integer_header_value(hdu.header, "NAXIS1", label)
    if row_width != hdu.columns.dtype.itemsize:
        raise ValueError(
            f"{label}: NAXIS1={row_width} is inconsistent with the "
            f"{hdu.columns.dtype.itemsize}-byte column layout."
        )
    for keyword, expected in (("FIRSTPIX", 0), ("LASTPIX", expected_npix - 1)):
        actual = integer_header_value(hdu.header, keyword, label)
        if actual != expected:
            raise ValueError(f"{label}: expected {keyword}={expected}, found {actual}.")

    validate_column_format(hdu.columns[PIXEL_COLUMN], "J", f"{label}:{PIXEL_COLUMN}")
    for column in COMPONENT_COLUMNS:
        validate_column_format(hdu.columns[column], "E", f"{label}:{column}")

    units = {
        column: parse_unit(hdu.columns[column].unit, f"{label}:{column}")
        for column in COMPONENT_COLUMNS
    }
    integrated_unit = u.GeV**2 / u.cm**5
    for column, unit in units.items():
        if not unit.is_equivalent(integrated_unit):
            raise ValueError(
                f"{label}:{column}: expected integrated J units equivalent "
                f"to {integrated_unit}, found {unit}."
            )
    reference_unit = units[SMOOTH_COLUMN]
    conversion_factors = {
        column: float(unit.to(reference_unit)) for column, unit in units.items()
    }

    ratio = nside // diagnostic_nside
    children_per_parent = ratio * ratio
    return MapMetadata(
        nside=nside,
        npix=expected_npix,
        children_per_parent=children_per_parent,
        coarse_npix=hp.nside2npix(diagnostic_nside),
        conversion_factors=conversion_factors,
    )


def nested_parent_pixels(
    pixels: np.ndarray,
    fine_nside: int,
    coarse_nside: int,
) -> np.ndarray:
    if not hp.isnsideok(fine_nside, nest=True) or not hp.isnsideok(
        coarse_nside, nest=True
    ):
        raise ValueError("Invalid fine or coarse NSIDE.")
    if fine_nside < coarse_nside or fine_nside % coarse_nside != 0:
        raise ValueError("fine NSIDE must be divisible by coarse NSIDE.")
    children_per_parent = (fine_nside // coarse_nside) ** 2
    pixels = np.asarray(pixels, dtype=np.int64)
    if np.any(pixels < 0) or np.any(pixels >= hp.nside2npix(fine_nside)):
        raise ValueError("Fine HEALPix pixel index is outside the valid range.")
    return pixels // children_per_parent


def _checked_component_chunk(
    table: fits.FITS_rec,
    column: str,
    start: int,
    stop: int,
    conversion_factor: float,
    path: Path,
) -> np.ndarray:
    values = np.asarray(table[column][start:stop], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}:{column}: non-finite value in rows {start}:{stop}.")
    if np.any(values < 0.0):
        raise ValueError(f"{path}:{column}: negative value in rows {start}:{stop}.")
    if conversion_factor != 1.0:
        values *= conversion_factor
    return values


def aggregate_map(
    path: Path,
    scenario: str,
    repop_id: int,
    input_nside: int,
    diagnostic_nside: int,
    chunk_size: int,
) -> CoarseRatioMap:
    """Stream one final map once and return sum-conserving coarse ratios."""
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Input FITS not found or empty: {path}")
    if chunk_size <= 0:
        raise ValueError("chunk-size must be a positive integer.")

    coarse_smooth: np.ndarray
    coarse_sub: np.ndarray
    fine_smooth_sum = 0.0
    fine_sub_sum = 0.0
    n_sentinel = 0

    with fits.open(path, memmap=True) as hdul:
        if HDU_NAME not in hdul:
            raise ValueError(f"{path}: missing {HDU_NAME!r} HDU.")
        hdu = hdul[HDU_NAME]
        metadata = validate_map_hdu_metadata(
            hdu,
            path,
            input_nside,
            diagnostic_nside,
        )
        table = hdu.data
        if table is None:
            raise ValueError(f"{path} [{HDU_NAME}]: HDU has no table data.")

        coarse_smooth = np.zeros(metadata.coarse_npix, dtype=np.float64)
        coarse_sub = np.zeros(metadata.coarse_npix, dtype=np.float64)
        coarse_counts = np.zeros(metadata.coarse_npix, dtype=np.int64)

        for start in range(0, metadata.npix, chunk_size):
            stop = min(start + chunk_size, metadata.npix)
            pixels = np.asarray(table[PIXEL_COLUMN][start:stop], dtype=np.int64)
            expected_pixels = np.arange(start, stop, dtype=np.int64)
            if not np.array_equal(pixels, expected_pixels):
                raise ValueError(
                    f"{path}: PIXEL must be the complete ordered NESTED "
                    f"sequence 0..{metadata.npix - 1}; mismatch in rows "
                    f"{start}:{stop}."
                )

            smooth = _checked_component_chunk(
                table,
                SMOOTH_COLUMN,
                start,
                stop,
                metadata.conversion_factors[SMOOTH_COLUMN],
                path,
            )
            pointlike = _checked_component_chunk(
                table,
                POINTLIKE_COLUMN,
                start,
                stop,
                metadata.conversion_factors[POINTLIKE_COLUMN],
                path,
            )
            extended_native = np.asarray(table[EXTENDED_COLUMN][start:stop])
            sentinel = extended_native == CLUMPY_EMPTY_JLIST
            n_sentinel += int(np.count_nonzero(sentinel))
            extended = np.asarray(extended_native, dtype=np.float64)
            if not np.all(np.isfinite(extended)):
                raise ValueError(
                    f"{path}:{EXTENDED_COLUMN}: non-finite value in rows "
                    f"{start}:{stop}."
                )
            if np.any(extended < 0.0):
                raise ValueError(
                    f"{path}:{EXTENDED_COLUMN}: negative value in rows "
                    f"{start}:{stop}."
                )
            extended = extended.copy()
            extended[sentinel] = 0.0
            extended_factor = metadata.conversion_factors[EXTENDED_COLUMN]
            if extended_factor != 1.0:
                extended *= extended_factor

            j_sub = pointlike + extended
            parents = nested_parent_pixels(
                pixels,
                input_nside,
                diagnostic_nside,
            )
            coarse_counts += np.bincount(parents, minlength=metadata.coarse_npix)
            coarse_smooth += np.bincount(
                parents,
                weights=smooth,
                minlength=metadata.coarse_npix,
            )
            coarse_sub += np.bincount(
                parents,
                weights=j_sub,
                minlength=metadata.coarse_npix,
            )
            fine_smooth_sum += float(smooth.sum(dtype=np.float64))
            fine_sub_sum += float(j_sub.sum(dtype=np.float64))

    expected_children = np.full(
        metadata.coarse_npix,
        metadata.children_per_parent,
        dtype=np.int64,
    )
    if not np.array_equal(coarse_counts, expected_children):
        raise RuntimeError(
            f"{path}: NESTED child counts do not cover every coarse pixel."
        )

    coarse_smooth_sum = float(coarse_smooth.sum(dtype=np.float64))
    coarse_sub_sum = float(coarse_sub.sum(dtype=np.float64))
    if not np.isclose(
        coarse_smooth_sum,
        fine_smooth_sum,
        rtol=SUM_RTOL,
        atol=0.0,
    ):
        raise RuntimeError(
            f"{path}: smooth aggregation did not conserve integrated J: "
            f"fine={fine_smooth_sum:.17g}, coarse={coarse_smooth_sum:.17g}."
        )
    if not np.isclose(
        coarse_sub_sum,
        fine_sub_sum,
        rtol=SUM_RTOL,
        atol=0.0,
    ):
        raise RuntimeError(
            f"{path}: substructure aggregation did not conserve integrated J: "
            f"fine={fine_sub_sum:.17g}, coarse={coarse_sub_sum:.17g}."
        )

    invalid_smooth = (~np.isfinite(coarse_smooth)) | (coarse_smooth <= 0.0)
    if np.any(invalid_smooth):
        bad = np.flatnonzero(invalid_smooth)
        preview = ", ".join(str(int(index)) for index in bad[:8])
        raise ValueError(
            f"{path}: non-positive or non-finite coarse Jsmooth in bin(s) "
            f"{preview}."
        )
    if np.any(~np.isfinite(coarse_sub)) or np.any(coarse_sub < 0.0):
        raise ValueError(f"{path}: invalid coarse J_sub values.")
    ratio = coarse_sub / coarse_smooth
    if not np.all(np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError(f"{path}: invalid coarse J_sub/J_smooth ratio.")

    return CoarseRatioMap(
        scenario=scenario,
        repop_id=repop_id,
        input_nside=input_nside,
        diagnostic_nside=diagnostic_nside,
        j_smooth=coarse_smooth,
        j_sub=coarse_sub,
        ratio=ratio,
        n_extended_sentinel_pixels=n_sentinel,
    )


def summarize_map(coarse_map: CoarseRatioMap) -> RatioSummary:
    ratio = np.asarray(coarse_map.ratio, dtype=np.float64)
    finite = ratio[np.isfinite(ratio)]
    if finite.size != ratio.size:
        raise ValueError(f"{coarse_map.scenario}: ratio contains non-finite bins.")
    n_bins = int(ratio.size)
    n_zero = int(np.count_nonzero(coarse_map.j_sub == 0.0))
    n_above = int(np.count_nonzero(ratio > 1.0))
    return RatioSummary(
        scenario=coarse_map.scenario,
        repop_id=coarse_map.repop_id,
        diagnostic_nside=coarse_map.diagnostic_nside,
        n_bins=n_bins,
        n_zero_substructure_bins=n_zero,
        min_finite_R=float(np.min(finite)),
        max_finite_R=float(np.max(finite)),
        n_bins_R_gt_1=n_above,
        fraction_bins_R_gt_1=n_above / n_bins,
    )


def prepare_log_ratio(ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ratio = np.asarray(ratio, dtype=np.float64)
    valid = np.isfinite(ratio) & (ratio > 0.0)
    result = np.full(ratio.shape, hp.UNSEEN, dtype=np.float64)
    result[valid] = np.log10(ratio[valid])
    return result, valid


def make_figure(
    maps: dict[str, CoarseRatioMap],
    repop_id: int,
    diagnostic_nside: int,
) -> plt.Figure:
    prepared = {
        scenario: prepare_log_ratio(maps[scenario].ratio) for scenario in SCENARIOS
    }
    norm = TwoSlopeNorm(
        vmin=PLOT_VMIN,
        vcenter=PLOT_VCENTER,
        vmax=PLOT_VMAX,
    )
    cmap_name = "coolwarm"
    figure = plt.figure(figsize=(14.0, 5.3))

    for index, scenario in enumerate(SCENARIOS, start=1):
        log_map, valid = prepared[scenario]
        display_map = log_map.copy()
        # Healpy reserves the colormap's under colour for the projection
        # background, so clip only this display copy to saturate low values.
        display_map[valid] = np.maximum(display_map[valid], PLOT_VMIN)
        hp.mollview(
            display_map,
            fig=figure.number,
            sub=(1, 2, index),
            nest=True,
            coord="G",
            title=f"{scenario.capitalize()} scenario",
            min=PLOT_VMIN,
            max=PLOT_VMAX,
            cmap=cmap_name,
            norm=norm,
            badcolor="#d9d9d9",
            bgcolor="white",
            cbar=False,
        )
        hp.graticule(dpar=30, dmer=30, alpha=0.25)

    scalar_mappable = ScalarMappable(
        norm=norm,
        cmap=plt.get_cmap(cmap_name),
    )
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(
        scalar_mappable,
        ax=figure.axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.10,
        aspect=45,
        extend="min",
        ticks=COLORBAR_TICKS,
    )
    colorbar.set_label(
        r"$\log_{10}(J_{\rm sub}/J_{\rm smooth})$",
        fontsize=13,
    )
    colorbar.ax.tick_params(labelsize=11)
    figure.suptitle(
        f"Repopulation {repop_id:04d}, diagnostic NSIDE={diagnostic_nside}",
        fontsize=14,
        y=0.98,
    )
    return figure


def write_summary_csv(path: Path, summaries: Sequence[RatioSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for summary in summaries:
            row = {column: getattr(summary, column) for column in SUMMARY_COLUMNS}
            for column in (
                "min_finite_R",
                "max_finite_R",
                "fraction_bins_R_gt_1",
            ):
                row[column] = format(row[column], ".17g")
            writer.writerow(row)


def print_summary(summary: RatioSummary) -> None:
    print(
        f"{summary.scenario}: repop_{summary.repop_id:04d}, "
        f"NSIDE={summary.diagnostic_nside}, bins={summary.n_bins}, "
        f"zero-substructure={summary.n_zero_substructure_bins}, "
        f"R_min={summary.min_finite_R:.8e}, "
        f"R_max={summary.max_finite_R:.8e}, "
        f"R>1={summary.n_bins_R_gt_1} "
        f"({summary.fraction_bins_R_gt_1:.6%})",
        flush=True,
    )
    if summary.n_zero_substructure_bins:
        print(
            f"WARNING: {summary.scenario} has "
            f"{summary.n_zero_substructure_bins} empty NSIDE="
            f"{summary.diagnostic_nside} substructure bin(s).",
            flush=True,
        )


def run_diagnostic(
    args: argparse.Namespace,
) -> tuple[dict[str, CoarseRatioMap], list[RatioSummary], list[Path]]:
    validate_cli_args(args)
    map_paths = resolve_map_paths(args)
    maps = {
        scenario: aggregate_map(
            path=map_paths[scenario],
            scenario=scenario,
            repop_id=args.repop_id,
            input_nside=args.input_nside,
            diagnostic_nside=args.diag_nside,
            chunk_size=args.chunk_size,
        )
        for scenario in SCENARIOS
    }
    summaries = [summarize_map(maps[scenario]) for scenario in SCENARIOS]
    for summary in summaries:
        print_summary(summary)

    output_dir = args.output_dir or default_output_dir(args.repop_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.basename or (
        f"coarse_subhalo_to_smooth_repop_{args.repop_id:04d}_" f"nside{args.diag_nside}"
    )
    summary_path = output_dir / f"{basename}_summary.csv"
    write_summary_csv(summary_path, summaries)

    figure = make_figure(maps, args.repop_id, args.diag_nside)
    saved = [summary_path]
    for output_format in dict.fromkeys(args.formats):
        output_path = output_dir / f"{basename}.{output_format}"
        save_kwargs = {"bbox_inches": "tight"}
        if output_format == "png":
            save_kwargs["dpi"] = 220
        figure.savefig(output_path, **save_kwargs)
        saved.append(output_path)
    plt.close(figure)
    return maps, summaries, saved


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _, _, saved = run_diagnostic(args)
    for path in saved:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
