#!/usr/bin/env python3
"""Plot Auriga repop_0000 templates with observed dwarfs added."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DWARF_MAPS = {
    "einasto": REPOSITORY_ROOT
    / "outputs/dwarfs/einasto/annihil_gal2D_LOS0_0_FOV360x180_nside2048.fits",
    "burkert": REPOSITORY_ROOT
    / "outputs/dwarfs/burkert/annihil_gal2D_LOS0_0_FOV360x180_nside2048.fits",
}
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/dwarfs/plots"
HDU_NAME = "JFACTOR"
PIXEL_COLUMN = "PIXEL"
AURIGA_COLUMN = "Jtot"
DWARF_COLUMN = "Jlist"
EXPECTED_NSIDE = 2048
EXPECTED_UNIT = u.GeV**2 / u.cm**5
# Exact float32 marker used by CLUMPY for an empty Jlist pixel.
CLUMPY_EMPTY_JLIST = np.float32(1.0e-40)
DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_XSIZE = 2000
PANEL_TITLE_FONTSIZE = 22.6
LABEL_FONTSIZE = 20.1
TICK_FONTSIZE = 18.9


@dataclass(frozen=True)
class MapMetadata:
    path: Path
    nside: int
    npix: int
    ordering: str
    coordsys: str
    unit: u.UnitBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add observed-dwarf Jlist maps to the fragile and resilient "
            "repop_0000 all-combined Auriga maps for visualization."
        )
    )
    parser.add_argument(
        "--fragile-map",
        type=Path,
        required=True,
        help="Current production fragile repop_0000 all-combined FITS file.",
    )
    parser.add_argument(
        "--resilient-map",
        type=Path,
        required=True,
        help="Current production resilient repop_0000 all-combined FITS file.",
    )
    parser.add_argument(
        "--einasto-map",
        type=Path,
        default=DEFAULT_DWARF_MAPS["einasto"],
    )
    parser.add_argument(
        "--burkert-map",
        type=Path,
        default=DEFAULT_DWARF_MAPS["burkert"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional lower log10(J/[GeV^2 cm^-5]) color limit.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional upper log10(J/[GeV^2 cm^-5]) color limit.",
    )
    parser.add_argument(
        "--xsize",
        type=int,
        default=DEFAULT_XSIZE,
        help="Horizontal Mollweide raster size per panel (default: 2000).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per processing chunk (default: 1000000).",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": LABEL_FONTSIZE,
            "axes.titlesize": PANEL_TITLE_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
        }
    )


def normalized_header_text(value: object) -> str:
    return str(value or "").strip().upper()


def open_validated_hdu(
    stack: ExitStack,
    path: Path,
    value_column: str,
) -> tuple[fits.BinTableHDU, MapMetadata]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"FITS file not found or empty: {path}")
    hdul = stack.enter_context(fits.open(path, mode="readonly", memmap=True))
    if HDU_NAME not in hdul:
        raise ValueError(f"{path}: missing {HDU_NAME!r} extension.")
    hdu = hdul[HDU_NAME]
    label = f"{path} [{HDU_NAME}]"
    if not isinstance(hdu, fits.BinTableHDU):
        raise ValueError(f"{label}: expected a binary table HDU.")
    missing = [
        name for name in (PIXEL_COLUMN, value_column) if name not in hdu.columns.names
    ]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}.")

    for keyword, expected in (
        ("PIXTYPE", "HEALPIX"),
        ("COORDSYS", "G"),
        ("INDXSCHM", "EXPLICIT"),
    ):
        actual = normalized_header_text(hdu.header.get(keyword))
        if actual != expected:
            raise ValueError(
                f"{label}: expected {keyword}={expected!r}, found {actual!r}."
            )
    ordering = normalized_header_text(hdu.header.get("ORDERING"))
    if ordering not in {"NEST", "NESTED"}:
        raise ValueError(f"{label}: expected NESTED ordering, found {ordering!r}.")

    try:
        nside = int(hdu.header["NSIDE"])
        nrows = int(hdu.header["NAXIS2"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: incomplete HEALPix metadata.") from error
    npix = hp.nside2npix(nside)
    if nside != EXPECTED_NSIDE or nrows != npix:
        raise ValueError(
            f"{label}: expected NSIDE={EXPECTED_NSIDE} and "
            f"{hp.nside2npix(EXPECTED_NSIDE):,} rows; found NSIDE={nside} "
            f"and {nrows:,} rows."
        )

    unit_text = hdu.columns[value_column].unit
    if unit_text is None or not str(unit_text).strip():
        raise ValueError(f"{label}:{value_column}: missing unit.")
    try:
        unit = u.Unit(unit_text, format="fits")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label}:{value_column}: invalid unit {unit_text!r}."
        ) from error
    if unit != EXPECTED_UNIT:
        raise ValueError(
            f"{label}:{value_column}: expected {EXPECTED_UNIT}, found {unit}."
        )

    metadata = MapMetadata(
        path=path,
        nside=nside,
        npix=npix,
        ordering="NESTED",
        coordsys="G",
        unit=unit,
    )
    return hdu, metadata


def require_matching_metadata(
    reference: MapMetadata,
    candidate: MapMetadata,
) -> None:
    reference_values = (
        reference.nside,
        reference.npix,
        reference.ordering,
        reference.coordsys,
        reference.unit,
    )
    candidate_values = (
        candidate.nside,
        candidate.npix,
        candidate.ordering,
        candidate.coordsys,
        candidate.unit,
    )
    if candidate_values != reference_values:
        raise ValueError(
            f"Incompatible maps: {reference.path} and {candidate.path}."
        )


def validate_pixel_chunk(
    table: fits.FITS_rec,
    path: Path,
    start: int,
    stop: int,
) -> None:
    pixels = np.asarray(table[PIXEL_COLUMN][start:stop], dtype=np.int64)
    if not np.array_equal(pixels, np.arange(start, stop, dtype=np.int64)):
        raise ValueError(
            f"{path}: PIXEL is not the complete ordered NESTED sequence "
            f"in rows {start}:{stop}."
        )


def checked_values(
    table: fits.FITS_rec,
    column: str,
    path: Path,
    start: int,
    stop: int,
    replace_clumpy_floor: bool = False,
) -> np.ndarray:
    values = np.asarray(table[column][start:stop], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            f"{path}:{column}: non-finite or negative value in rows "
            f"{start}:{stop}."
        )
    if replace_clumpy_floor:
        empty = values == float(CLUMPY_EMPTY_JLIST)
        if np.any(empty):
            values = values.copy()
            values[empty] = 0.0
    return values


def scan_scenario(
    auriga_path: Path,
    dwarf_paths: dict[str, Path],
    chunk_size: int,
) -> dict[str, tuple[float, float, int]]:
    statistics = {
        profile: [np.inf, -np.inf, 0] for profile in dwarf_paths
    }
    with ExitStack() as stack:
        auriga_hdu, auriga_metadata = open_validated_hdu(
            stack, auriga_path, AURIGA_COLUMN
        )
        dwarf_hdus = {}
        dwarf_metadata = {}
        for profile, path in dwarf_paths.items():
            hdu, metadata = open_validated_hdu(stack, path, DWARF_COLUMN)
            require_matching_metadata(auriga_metadata, metadata)
            dwarf_hdus[profile] = hdu
            dwarf_metadata[profile] = metadata

        auriga_table = auriga_hdu.data
        dwarf_tables = {profile: hdu.data for profile, hdu in dwarf_hdus.items()}
        if auriga_table is None or any(table is None for table in dwarf_tables.values()):
            raise ValueError("One or more JFACTOR extensions contain no table data.")

        for start in range(0, auriga_metadata.npix, chunk_size):
            stop = min(start + chunk_size, auriga_metadata.npix)
            validate_pixel_chunk(auriga_table, auriga_metadata.path, start, stop)
            auriga_values = checked_values(
                auriga_table,
                AURIGA_COLUMN,
                auriga_metadata.path,
                start,
                stop,
            )
            for profile, dwarf_table in dwarf_tables.items():
                metadata = dwarf_metadata[profile]
                validate_pixel_chunk(dwarf_table, metadata.path, start, stop)
                dwarf_values = checked_values(
                    dwarf_table,
                    DWARF_COLUMN,
                    metadata.path,
                    start,
                    stop,
                    replace_clumpy_floor=True,
                )
                combined = auriga_values + dwarf_values
                positive = combined > 0.0
                count = int(np.count_nonzero(positive))
                statistics[profile][2] += count
                if count:
                    log_values = np.log10(combined[positive])
                    statistics[profile][0] = min(
                        statistics[profile][0], float(log_values.min())
                    )
                    statistics[profile][1] = max(
                        statistics[profile][1], float(log_values.max())
                    )

    result = {}
    for profile, (minimum, maximum, n_positive) in statistics.items():
        if n_positive == 0:
            raise ValueError(f"{profile}: combined map has no positive pixels.")
        result[profile] = (float(minimum), float(maximum), int(n_positive))
    return result


def load_combined_log_map(
    auriga_table: fits.FITS_rec,
    dwarf_table: fits.FITS_rec,
    npix: int,
    chunk_size: int,
) -> np.ndarray:
    log_map = np.full(npix, hp.UNSEEN, dtype=np.float32)
    for start in range(0, npix, chunk_size):
        stop = min(start + chunk_size, npix)
        auriga_values = np.asarray(
            auriga_table[AURIGA_COLUMN][start:stop], dtype=np.float64
        )
        dwarf_values = np.asarray(
            dwarf_table[DWARF_COLUMN][start:stop], dtype=np.float64
        )
        empty = dwarf_values == float(CLUMPY_EMPTY_JLIST)
        if np.any(empty):
            dwarf_values = dwarf_values.copy()
            dwarf_values[empty] = 0.0
        combined = auriga_values + dwarf_values
        positive = combined > 0.0
        destination = log_map[start:stop]
        destination[positive] = np.log10(combined[positive]).astype(
            np.float32, copy=False
        )
    return log_map


def determine_limits(
    all_statistics: dict[str, dict[str, tuple[float, float, int]]],
    manual_vmin: float | None,
    manual_vmax: float | None,
) -> tuple[float, float]:
    ranges = [
        statistics[profile][:2]
        for statistics in all_statistics.values()
        for profile in ("einasto", "burkert")
    ]
    vmin = min(item[0] for item in ranges) if manual_vmin is None else manual_vmin
    vmax = max(item[1] for item in ranges) if manual_vmax is None else manual_vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Color limits must be finite and satisfy vmin < vmax.")
    return float(vmin), float(vmax)


def make_scenario_figure(
    scenario: str,
    auriga_path: Path,
    dwarf_paths: dict[str, Path],
    vmin: float,
    vmax: float,
    xsize: int,
    chunk_size: int,
) -> plt.Figure:
    configure_plot_style()
    figure = plt.figure(figsize=(15.364, 5.8), facecolor="white")
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#000000")

    with ExitStack() as stack:
        auriga_hdu, auriga_metadata = open_validated_hdu(
            stack, auriga_path, AURIGA_COLUMN
        )
        panels = (
            ("einasto", r"(a) Einasto"),
            ("burkert", r"(b) Burkert"),
        )
        for index, (profile, title) in enumerate(panels, start=1):
            dwarf_hdu, dwarf_metadata = open_validated_hdu(
                stack, dwarf_paths[profile], DWARF_COLUMN
            )
            require_matching_metadata(auriga_metadata, dwarf_metadata)
            log_map = load_combined_log_map(
                auriga_hdu.data,
                dwarf_hdu.data,
                auriga_metadata.npix,
                chunk_size,
            )
            hp.mollview(
                log_map,
                fig=figure.number,
                sub=(1, 2, index),
                nest=True,
                coord="G",
                flip="astro",
                xsize=xsize,
                title=title,
                min=vmin,
                max=vmax,
                cmap=cmap,
                badcolor="#000000",
                bgcolor="white",
                cbar=False,
                notext=True,
            )
            hp.graticule(dpar=30, dmer=30, color="#b8b8b8", alpha=0.25)
            del log_map

    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=figure.axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.10,
        aspect=50,
    )
    colorbar.set_label(
        r"$\log_{10}(J_{\rm Auriga+dwarfs}\,[\mathrm{GeV}^2\,"
        r"\mathrm{cm}^{-5}])$",
        labelpad=4.0,
    )
    colorbar.ax.tick_params(length=3.5, labelsize=TICK_FONTSIZE)
    figure.suptitle(
        f"{scenario.capitalize()} scenario, repopulation 0000",
        fontsize=PANEL_TITLE_FONTSIZE,
        y=0.98,
    )
    return figure


def save_figure(figure: plt.Figure, output_dir: Path, basename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("png", 300), ("pdf", 500)):
        output_path = output_dir / f"{basename}.{suffix}"
        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved: {output_path}", flush=True)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.xsize < 4 or args.xsize % 2:
        raise ValueError("--xsize must be an even integer of at least 4.")

    scenario_paths = {
        "fragile": args.fragile_map,
        "resilient": args.resilient_map,
    }
    dwarf_paths = {
        "einasto": args.einasto_map,
        "burkert": args.burkert_map,
    }
    all_statistics = {}
    for scenario, auriga_path in scenario_paths.items():
        statistics = scan_scenario(auriga_path, dwarf_paths, args.chunk_size)
        all_statistics[scenario] = statistics
        for profile, (minimum, maximum, n_positive) in statistics.items():
            print(
                f"{scenario} + {profile}: positive={n_positive:,}, "
                f"log10(J)=[{minimum:.8g}, {maximum:.8g}]",
                flush=True,
            )

    vmin, vmax = determine_limits(all_statistics, args.vmin, args.vmax)
    print(f"Shared color limits: [{vmin:.8g}, {vmax:.8g}]", flush=True)

    for scenario, auriga_path in scenario_paths.items():
        figure = make_scenario_figure(
            scenario,
            auriga_path,
            dwarf_paths,
            vmin,
            vmax,
            args.xsize,
            args.chunk_size,
        )
        save_figure(
            figure,
            args.output_dir,
            f"{scenario}_repop_0000_with_dwarfs",
        )


if __name__ == "__main__":
    main()
