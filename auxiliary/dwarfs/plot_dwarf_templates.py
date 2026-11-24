#!/usr/bin/env python3
"""Plot the Einasto and Burkert dwarf-only CLUMPY templates."""

from __future__ import annotations

import argparse
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
DEFAULT_MAPS = {
    "einasto": REPOSITORY_ROOT
    / "outputs/dwarfs/einasto/annihil_gal2D_LOS0_0_FOV360x180_nside2048.fits",
    "burkert": REPOSITORY_ROOT
    / "outputs/dwarfs/burkert/annihil_gal2D_LOS0_0_FOV360x180_nside2048.fits",
}
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/dwarfs/plots"
HDU_NAME = "JFACTOR"
PIXEL_COLUMN = "PIXEL"
J_COLUMN = "Jlist"
EXPECTED_NSIDE = 2048
EXPECTED_UNIT = u.GeV**2 / u.cm**5
# Exact float32 marker used by CLUMPY for an empty Jlist pixel.
CLUMPY_EMPTY_JLIST = np.float32(1.0e-40)
CHUNK_SIZE = 1_000_000
XSIZE = 2000
PANEL_TITLE_FONTSIZE = 22.6
LABEL_FONTSIZE = 20.1
TICK_FONTSIZE = 18.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a shared-scale two-panel Mollweide figure of the "
            "Einasto and Burkert dwarf-only J-factor templates."
        )
    )
    parser.add_argument("--einasto-map", type=Path, default=DEFAULT_MAPS["einasto"])
    parser.add_argument("--burkert-map", type=Path, default=DEFAULT_MAPS["burkert"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--basename",
        default="dwarf_only_templates",
        help="Output basename without an extension (default: dwarf_only_templates).",
    )
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
        default=XSIZE,
        help="Horizontal Mollweide raster size per panel (default: 2000).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Rows per chunk while validating and finding limits.",
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


def validate_hdu(hdu: fits.BinTableHDU, path: Path) -> int:
    label = f"{path} [{HDU_NAME}]"
    if not isinstance(hdu, fits.BinTableHDU):
        raise ValueError(f"{label}: expected a binary table HDU.")
    missing = [
        name for name in (PIXEL_COLUMN, J_COLUMN) if name not in hdu.columns.names
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

    unit_text = hdu.columns[J_COLUMN].unit
    if unit_text is None or not str(unit_text).strip():
        raise ValueError(f"{label}:{J_COLUMN}: missing unit.")
    try:
        unit = u.Unit(unit_text, format="fits")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}:{J_COLUMN}: invalid unit {unit_text!r}.") from error
    if unit != EXPECTED_UNIT:
        raise ValueError(
            f"{label}:{J_COLUMN}: expected {EXPECTED_UNIT}, found {unit}."
        )
    return npix


def scan_map(path: Path, chunk_size: int) -> tuple[float, float, int, int]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"FITS file not found or empty: {path}")

    minimum = np.inf
    maximum = -np.inf
    n_positive = 0
    n_empty = 0
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        if HDU_NAME not in hdul:
            raise ValueError(f"{path}: missing {HDU_NAME!r} extension.")
        hdu = hdul[HDU_NAME]
        npix = validate_hdu(hdu, path)
        table = hdu.data
        if table is None:
            raise ValueError(f"{path} [{HDU_NAME}]: missing table data.")

        for start in range(0, npix, chunk_size):
            stop = min(start + chunk_size, npix)
            pixels = np.asarray(table[PIXEL_COLUMN][start:stop], dtype=np.int64)
            if not np.array_equal(pixels, np.arange(start, stop, dtype=np.int64)):
                raise ValueError(
                    f"{path}: PIXEL is not the complete ordered NESTED sequence "
                    f"in rows {start}:{stop}."
                )
            values = np.asarray(table[J_COLUMN][start:stop])
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(
                    f"{path}:{J_COLUMN}: non-finite or negative value in rows "
                    f"{start}:{stop}."
                )
            empty = (values == 0.0) | (values == CLUMPY_EMPTY_JLIST)
            positive = ~empty & (values > 0.0)
            n_positive += int(np.count_nonzero(positive))
            n_empty += int(np.count_nonzero(empty))
            if np.any(positive):
                log_values = np.log10(values[positive].astype(np.float64))
                minimum = min(minimum, float(log_values.min()))
                maximum = max(maximum, float(log_values.max()))

    if n_positive == 0:
        raise ValueError(f"{path}:{J_COLUMN}: no strictly positive pixels.")
    return minimum, maximum, n_positive, n_empty


def load_log_map(path: Path, chunk_size: int) -> np.ndarray:
    """Return a display copy; the memory-mapped FITS values are never modified."""
    path = path.expanduser().resolve()
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        hdu = hdul[HDU_NAME]
        npix = validate_hdu(hdu, path)
        values = hdu.data[J_COLUMN]
        log_map = np.full(npix, hp.UNSEEN, dtype=np.float32)
        for start in range(0, npix, chunk_size):
            stop = min(start + chunk_size, npix)
            chunk = np.asarray(values[start:stop])
            empty = (chunk == 0.0) | (chunk == CLUMPY_EMPTY_JLIST)
            positive = ~empty & (chunk > 0.0)
            destination = log_map[start:stop]
            destination[positive] = np.log10(chunk[positive]).astype(
                np.float32, copy=False
            )
    return log_map


def determine_limits(
    ranges: list[tuple[float, float]],
    manual_vmin: float | None,
    manual_vmax: float | None,
) -> tuple[float, float]:
    vmin = min(item[0] for item in ranges) if manual_vmin is None else manual_vmin
    vmax = max(item[1] for item in ranges) if manual_vmax is None else manual_vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Color limits must be finite and satisfy vmin < vmax.")
    return float(vmin), float(vmax)


def make_figure(
    paths: dict[str, Path],
    vmin: float,
    vmax: float,
    xsize: int,
    chunk_size: int,
) -> plt.Figure:
    configure_plot_style()
    figure = plt.figure(figsize=(15.364, 5.8), facecolor="white")
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#000000")

    panels = (
        ("einasto", r"(a) Einasto"),
        ("burkert", r"(b) Burkert"),
    )
    for index, (profile, title) in enumerate(panels, start=1):
        log_map = load_log_map(paths[profile], chunk_size)
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
        projection_cmap = plt.gca().get_images()[0].cmap
        projection_cmap.set_under("#ffffff")
        projection_cmap.set_bad("#000000")
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
        r"$\log_{10}(J_{\rm dwarfs}\,[\mathrm{GeV}^2\,\mathrm{cm}^{-5}])$",
        labelpad=4.0,
    )
    colorbar.ax.tick_params(length=3.5, labelsize=TICK_FONTSIZE)
    return figure


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.xsize < 4 or args.xsize % 2:
        raise ValueError("--chunk-size must be positive and --xsize a positive even integer.")
    if Path(args.basename).name != args.basename:
        raise ValueError("--basename must not contain a directory component.")

    paths = {
        "einasto": args.einasto_map,
        "burkert": args.burkert_map,
    }
    ranges = []
    for profile, path in paths.items():
        minimum, maximum, n_positive, n_empty = scan_map(path, args.chunk_size)
        ranges.append((minimum, maximum))
        print(
            f"{profile}: positive={n_positive:,}, empty={n_empty:,}, "
            f"log10(J)=[{minimum:.8g}, {maximum:.8g}]",
            flush=True,
        )

    vmin, vmax = determine_limits(ranges, args.vmin, args.vmax)
    print(f"Shared color limits: [{vmin:.8g}, {vmax:.8g}]", flush=True)
    figure = make_figure(paths, vmin, vmax, args.xsize, args.chunk_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("png", 300), ("pdf", 500)):
        output_path = args.output_dir / f"{args.basename}.{suffix}"
        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved: {output_path}", flush=True)
    plt.close(figure)


if __name__ == "__main__":
    main()
