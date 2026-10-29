#!/usr/bin/env python3
"""Experimental conservative-sum Mollweide projection for Auriga--CLUMPY.

Unlike the conventional center-sampled projection, this prototype projects
every HEALPix pixel center exactly once and sums integrated J-factors in the
assigned Mollweide raster cell.  It does not replace the conventional plotter.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import healpy as hp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from scipy.spatial import cKDTree


SCENARIOS = ("fragile", "resilient")
HDU_NAME = "JFACTOR"
PIXEL_COLUMN = "PIXEL"
DEFAULT_NSIDE = 2048
DEFAULT_XSIZE = 800
DEFAULT_CHUNK_SIZE = 1_000_000
SUPPORTED_COLORMAPS = ("magma", "plasma", "turbo", "viridis")
SUPPORTED_OUTPUT_FORMATS = ("png", "pdf")
EXPECTED_UNIT = u.GeV**2 / u.cm**5
CLUMPY_EMPTY_JEXTENDED = np.float32(1.0e-40)
ZERO_SENTINEL = np.float32(-np.finfo(np.float32).max)
# Compensate for scaling the 15.364-inch PDF to a 440 TeX pt text width.
PANEL_TITLE_FONTSIZE = 26.4
LABEL_FONTSIZE = 23.9
TICK_FONTSIZE = 21.4
LEGEND_FONTSIZE = 21.4


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
            "legend.fontsize": LEGEND_FONTSIZE,
        }
    )


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    column: str
    title: str


COMPONENTS = (
    ComponentSpec("smooth", "Jsmooth", "Smooth MW halo"),
    ComponentSpec("pointlike", "Jpointlike", "Pointlike subhalos"),
    ComponentSpec("extended", "Jextended", "Extended subhalos"),
    ComponentSpec("total", "Jtot", "Final template"),
)


@dataclass(frozen=True)
class FitsMetadata:
    nside: int
    npix: int
    conversion_factors: dict[str, float]


@dataclass
class ComponentStatistics:
    n_zero: int = 0
    n_positive: int = 0
    n_invalid: int = 0
    n_empty_extended_sentinel: int = 0
    min_positive_j: float = math.inf
    max_j: float = -math.inf
    min_log10_j: float = math.inf
    max_log10_j: float = -math.inf


@dataclass(frozen=True)
class ProjectionGeometry:
    xsize: int
    ysize: int
    inside: np.ndarray
    healpix_pixels: np.ndarray


def expected_map_path(
    maps_root: Path,
    scenario: str,
    repop_id: int,
    nside: int,
) -> Path:
    return (
        maps_root
        / scenario
        / "outputs"
        / "total"
        / f"repop_{repop_id:04d}_nside{nside}"
        / f"auriga_total_nside{nside}.fits"
    )


def resolve_map_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {}
    for scenario in SCENARIOS:
        override = getattr(args, f"{scenario}_map")
        path = (
            override
            if override is not None
            else expected_map_path(
                args.maps_root,
                scenario,
                args.repop_id,
                args.nside,
            )
        )
        path = Path(path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{scenario}: FITS not found or empty: {path}")
        paths[scenario] = path
    return paths


def normalized_header_text(value: object) -> str:
    return str(value or "").strip().upper()


def validate_hdu(
    hdu: fits.BinTableHDU,
    path: Path,
    expected_nside: int,
) -> FitsMetadata:
    label = f"{path} [{HDU_NAME}]"
    if not isinstance(hdu, fits.BinTableHDU):
        raise ValueError(f"{label}: expected a BINTABLE HDU.")
    required_columns = (PIXEL_COLUMN,) + tuple(
        component.column for component in COMPONENTS
    )
    missing = [name for name in required_columns if name not in hdu.columns.names]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}.")

    for keyword, expected in (
        ("PIXTYPE", "HEALPIX"),
        ("INDXSCHM", "EXPLICIT"),
        ("COORDSYS", "G"),
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
        n_rows = int(hdu.header["NAXIS2"])
        first_pixel = int(hdu.header["FIRSTPIX"])
        last_pixel = int(hdu.header["LASTPIX"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: incomplete integer HEALPix metadata.") from error
    if nside != expected_nside:
        raise ValueError(f"{label}: expected NSIDE={expected_nside}, found {nside}.")
    npix = hp.nside2npix(nside)
    if n_rows != npix or first_pixel != 0 or last_pixel != npix - 1:
        raise ValueError(
            f"{label}: incomplete full-sky table metadata for {npix} pixels."
        )

    conversion_factors = {}
    for component in COMPONENTS:
        unit_text = hdu.columns[component.column].unit
        if unit_text is None or not str(unit_text).strip():
            raise ValueError(f"{label}:{component.column}: missing unit.")
        try:
            unit = u.Unit(unit_text, format="fits")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label}:{component.column}: invalid unit {unit_text!r}."
            ) from error
        if not unit.is_equivalent(EXPECTED_UNIT):
            raise ValueError(
                f"{label}:{component.column}: expected integrated J units, "
                f"found {unit}."
            )
        conversion_factors[component.key] = float(unit.to(EXPECTED_UNIT))
    return FitsMetadata(
        nside=nside,
        npix=npix,
        conversion_factors=conversion_factors,
    )


def checked_values(
    table: fits.FITS_rec,
    component: ComponentSpec,
    selector: slice | np.ndarray,
    conversion_factor: float,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(table[component.column][selector], dtype=np.float64)
    empty_sentinel = np.zeros(values.shape, dtype=np.bool_)
    if component.key == "extended":
        empty_sentinel = values == float(CLUMPY_EMPTY_JEXTENDED)
    if conversion_factor != 1.0:
        with np.errstate(over="ignore", invalid="ignore"):
            values *= conversion_factor
    return values, empty_sentinel


def value_masks(
    values: np.ndarray,
    empty_sentinel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    invalid = ~np.isfinite(values) | (values < 0.0)
    positive = ~invalid & ~empty_sentinel & (values > 0.0)
    zero = ~invalid & ((values == 0.0) | empty_sentinel)
    return positive, zero, invalid


def scan_value_ranges(
    paths: dict[str, Path],
    expected_nside: int,
    chunk_size: int,
) -> tuple[dict[str, dict[str, ComponentStatistics]], dict[str, FitsMetadata]]:
    statistics = {
        scenario: {component.key: ComponentStatistics() for component in COMPONENTS}
        for scenario in SCENARIOS
    }
    metadata_by_scenario = {}

    for scenario in SCENARIOS:
        path = paths[scenario]
        print(f"Scanning value ranges: {scenario}: {path}", flush=True)
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            if HDU_NAME not in hdul:
                raise ValueError(f"{path}: missing {HDU_NAME!r} HDU.")
            hdu = hdul[HDU_NAME]
            metadata = validate_hdu(hdu, path, expected_nside)
            metadata_by_scenario[scenario] = metadata
            table = hdu.data
            if table is None:
                raise ValueError(f"{path} [{HDU_NAME}]: missing table data.")

            for start in range(0, metadata.npix, chunk_size):
                stop = min(start + chunk_size, metadata.npix)
                pixels = np.asarray(table[PIXEL_COLUMN][start:stop], dtype=np.int64)
                if not np.array_equal(pixels, np.arange(start, stop, dtype=np.int64)):
                    raise ValueError(
                        f"{path}: PIXEL is not the complete ordered NESTED "
                        f"sequence in rows {start}:{stop}."
                    )

                for component in COMPONENTS:
                    values, empty_sentinel = checked_values(
                        table,
                        component,
                        slice(start, stop),
                        metadata.conversion_factors[component.key],
                        path,
                    )
                    stats = statistics[scenario][component.key]
                    stats.n_empty_extended_sentinel += int(
                        np.count_nonzero(empty_sentinel)
                    )
                    positive, zero, invalid = value_masks(values, empty_sentinel)
                    stats.n_zero += int(np.count_nonzero(zero))
                    stats.n_invalid += int(np.count_nonzero(invalid))
                    n_positive = int(np.count_nonzero(positive))
                    stats.n_positive += n_positive
                    if n_positive:
                        positive_values = values[positive]
                        log_values = np.log10(positive_values)
                        stats.min_positive_j = min(
                            stats.min_positive_j,
                            float(positive_values.min()),
                        )
                        stats.max_j = max(stats.max_j, float(positive_values.max()))
                        stats.min_log10_j = min(
                            stats.min_log10_j,
                            float(log_values.min()),
                        )
                        stats.max_log10_j = max(
                            stats.max_log10_j,
                            float(log_values.max()),
                        )
                print(
                    f"  {scenario}: rows {start:,}:{stop:,} / {metadata.npix:,}",
                    flush=True,
                )

    for component in COMPONENTS:
        if not any(
            statistics[scenario][component.key].n_positive
            for scenario in SCENARIOS
        ):
            raise ValueError(f"{component.column}: no positive values in either map.")
    return statistics, metadata_by_scenario


def make_projection_geometry(nside: int, xsize: int) -> ProjectionGeometry:
    projector = hp.projector.MollweideProj(
        coord="G",
        xsize=xsize,
        flipconv="astro",
    )
    x, y = projector.ij2xy()
    x_data = np.asarray(x)
    y_data = np.asarray(y)
    inside = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
    theta, phi = projector.xy2ang(x_data[inside], y_data[inside], lonlat=False)
    pixels = hp.ang2pix(nside, theta, phi, nest=True).astype(np.int64)
    return ProjectionGeometry(
        xsize=xsize,
        ysize=xsize // 2,
        inside=inside,
        healpix_pixels=pixels,
    )


def add_graticule(
    axes: plt.Axes,
    projector: hp.projector.MollweideProj,
    color: str,
) -> None:
    longitude = np.linspace(-180.0, 180.0, 1441)
    latitude = np.linspace(-89.999, 89.999, 721)
    for latitude_deg in (-60.0, -30.0, 0.0, 30.0, 60.0):
        x, y = projector.ang2xy(
            longitude,
            np.full(longitude.shape, latitude_deg),
            lonlat=True,
        )
        axes.plot(x, y, color=color, alpha=0.20, linewidth=0.45)
    for longitude_deg in range(-150, 181, 30):
        x, y = projector.ang2xy(
            np.full(latitude.shape, float(longitude_deg)),
            latitude,
            lonlat=True,
        )
        axes.plot(x, y, color=color, alpha=0.20, linewidth=0.45)


def pixel_layout(
    xsize: int,
    panel_scale: float,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    panel_width = max(1, int(round(xsize * panel_scale)))
    panel_height = max(1, int(round((xsize // 2) * panel_scale)))
    left = 60
    right = 60
    top = 90
    bottom = 260
    column_gap = 60
    row_gap = 375
    colorbar_bottom = 180
    colorbar_gap = bottom - colorbar_bottom
    colorbar_height = 30

    canvas_width = left + 2 * panel_width + column_gap + right
    canvas_height = top + 2 * panel_height + row_gap + bottom
    panel_origins = []
    for panel_index in range(4):
        row, column = divmod(panel_index, 2)
        panel_origins.append(
            (
                left + column * (panel_width + column_gap),
                bottom + (1 - row) * (panel_height + row_gap),
            )
        )

    colorbar_rectangles = []
    colorbar_width = int(round(0.78 * panel_width))
    for panel_left, panel_bottom in panel_origins:
        colorbar_left = panel_left + (panel_width - colorbar_width) // 2
        colorbar_rectangles.append(
            (
                colorbar_left,
                panel_bottom - colorbar_gap,
                colorbar_width,
                colorbar_height,
            )
        )
    return canvas_width, canvas_height, panel_origins, colorbar_rectangles


def rectangle_fraction(
    rectangle: tuple[int, int, int, int],
    canvas_width: int,
    canvas_height: int,
) -> tuple[float, float, float, float]:
    left, bottom, width, height = rectangle
    return (
        left / canvas_width,
        bottom / canvas_height,
        width / canvas_width,
        height / canvas_height,
    )


def display_colormap(name: str) -> matplotlib.colors.Colormap:
    cmap = matplotlib.colormaps[name].copy()
    cmap.set_under(cmap(0.0))
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    return cmap


def render_figure(
    scenario: str,
    projected: dict[str, dict[str, np.ndarray]],
    limits: dict[str, tuple[float, float]],
    output_path: Path,
    args: argparse.Namespace,
) -> tuple[int, int]:
    configure_plot_style()
    (
        canvas_width,
        canvas_height,
        panel_origins,
        colorbar_rectangles,
    ) = pixel_layout(args.xsize, args.panel_scale)
    panel_width = int(round(args.xsize * args.panel_scale))
    panel_height = int(round((args.xsize // 2) * args.panel_scale))

    figure = plt.figure(
        figsize=(canvas_width / args.dpi, canvas_height / args.dpi),
        dpi=args.dpi,
        facecolor=args.outside_color,
    )
    projector = hp.projector.MollweideProj(
        coord="G",
        xsize=args.xsize,
        flipconv="astro",
    )
    zero_cmap = ListedColormap([args.zero_color])
    base_cmap = display_colormap(args.cmap)

    for panel_index, component in enumerate(COMPONENTS):
        origin = panel_origins[panel_index]
        axes = figure.add_axes(
            rectangle_fraction(
                (origin[0], origin[1], panel_width, panel_height),
                canvas_width,
                canvas_height,
            )
        )
        axes.set_facecolor(args.outside_color)
        raster = projected[scenario][component.key]
        inside = np.isfinite(raster)
        positive = inside & (raster != ZERO_SENTINEL)

        zero_background = np.ma.array(
            np.zeros(raster.shape, dtype=np.uint8),
            mask=~inside,
        )
        axes.imshow(
            zero_background,
            origin="lower",
            extent=(-2.0, 2.0, -1.0, 1.0),
            interpolation="nearest",
            cmap=zero_cmap,
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        vmin, vmax = limits[component.key]
        positive_layer = np.ma.array(
            np.where(positive, raster, vmin),
            mask=~positive,
        )
        axes.imshow(
            positive_layer,
            origin="lower",
            extent=(-2.0, 2.0, -1.0, 1.0),
            interpolation="nearest",
            cmap=base_cmap,
            norm=Normalize(vmin=vmin, vmax=vmax, clip=False),
            rasterized=True,
        )
        if not args.no_graticule:
            add_graticule(axes, projector, args.graticule_color)
        axes.set_xlim(-2.0, 2.0)
        axes.set_ylim(-1.0, 1.0)
        axes.set_aspect("equal")
        axes.axis("off")

    for panel_index, component in enumerate(COMPONENTS):
        panel_left, panel_bottom = panel_origins[panel_index]
        center_x = (panel_left + panel_width / 2.0) / canvas_width
        title_y = (panel_bottom + panel_height + 25) / canvas_height
        figure.text(
            center_x,
            title_y,
            component.title,
            ha="center",
            va="bottom",
            fontsize=matplotlib.rcParams["axes.titlesize"],
            fontweight="semibold",
        )

        colorbar_axes = figure.add_axes(
            rectangle_fraction(
                colorbar_rectangles[panel_index],
                canvas_width,
                canvas_height,
            )
        )
        vmin, vmax = limits[component.key]
        normalization = Normalize(vmin=vmin, vmax=vmax, clip=False)
        colorbar = figure.colorbar(
            ScalarMappable(norm=normalization, cmap=base_cmap),
            cax=colorbar_axes,
            orientation="horizontal",
            extend="neither",
        )
        colorbar.ax.tick_params(length=3.5, labelsize=TICK_FONTSIZE)
        colorbar.set_label(
            r"$\log_{10}(J\,[\mathrm{GeV}^2\,\mathrm{cm}^{-5}])$",
            labelpad=2,
        )

    figure.legend(
        handles=[
            Patch(facecolor=args.zero_color, edgecolor="none", label=r"$J = 0$")
        ],
        loc="lower left",
        bbox_to_anchor=(16 / canvas_width, 16 / canvas_height),
        bbox_transform=figure.transFigure,
        frameon=False,
    )
    output_dpi = (
        args.pdf_dpi if output_path.suffix.lower() == ".pdf" else args.dpi
    )
    figure.savefig(
        output_path,
        dpi=output_dpi,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    return canvas_width, canvas_height


def serializable_statistics(
    statistics: dict[str, dict[str, ComponentStatistics]],
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    result = {}
    for scenario in SCENARIOS:
        result[scenario] = {}
        for component in COMPONENTS:
            payload = asdict(statistics[scenario][component.key])
            if not statistics[scenario][component.key].n_positive:
                payload["min_positive_j"] = None
                payload["max_j"] = None
                payload["min_log10_j"] = None
                payload["max_log10_j"] = None
            result[scenario][component.key] = payload
    return result


# Keep the conservative implementation below unchanged while making this
# committed script independent of the unversioned exploratory plotter.
conventional = SimpleNamespace(
    DEFAULT_NSIDE=DEFAULT_NSIDE,
    DEFAULT_XSIZE=DEFAULT_XSIZE,
    DEFAULT_CHUNK_SIZE=DEFAULT_CHUNK_SIZE,
    SUPPORTED_COLORMAPS=SUPPORTED_COLORMAPS,
    SCENARIOS=SCENARIOS,
    COMPONENTS=COMPONENTS,
    HDU_NAME=HDU_NAME,
    PIXEL_COLUMN=PIXEL_COLUMN,
    ZERO_SENTINEL=ZERO_SENTINEL,
    FitsMetadata=FitsMetadata,
    ComponentStatistics=ComponentStatistics,
    resolve_map_paths=resolve_map_paths,
    checked_values=checked_values,
    value_masks=value_masks,
    scan_value_ranges=scan_value_ranges,
    make_projection_geometry=make_projection_geometry,
    render_figure=render_figure,
    serializable_statistics=serializable_statistics,
)


@dataclass(frozen=True)
class ConservativeGeometry:
    projector: hp.projector.MollweideProj
    xsize: int
    ysize: int
    inside: np.ndarray
    valid_flat_cells: np.ndarray
    valid_cell_xy: np.ndarray
    valid_cell_tree: cKDTree


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental conservative-sum projection of fragile/resilient "
            "Auriga--CLUMPY components into 2x2 Mollweide figures."
        )
    )
    parser.add_argument("--repop-id", type=int, default=0)
    parser.add_argument("--nside", type=int, default=conventional.DEFAULT_NSIDE)
    parser.add_argument(
        "--maps-root",
        type=Path,
        default=Path("outputs/clumpy"),
    )
    parser.add_argument("--fragile-map", type=Path, default=None)
    parser.add_argument("--resilient-map", type=Path, default=None)
    parser.add_argument("--xsize", type=int, default=conventional.DEFAULT_XSIZE)
    parser.add_argument("--panel-scale", type=float, default=2.0)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=SUPPORTED_OUTPUT_FORMATS,
        default=("png",),
        help="Figure formats to write (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help=(
            "PNG resolution and reference layout DPI; retaining the default "
            "preserves the approved PNG geometry (default: 220)."
        ),
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=500,
        help=(
            "Rasterization DPI for map layers embedded directly in PDF "
            "outputs (default: 500)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=conventional.DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--normalization",
        choices=("percentile", "full"),
        default="full",
        help="Normalization of aggregated projected cells (default: full).",
    )
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.9)
    parser.add_argument(
        "--cmap",
        choices=conventional.SUPPORTED_COLORMAPS,
        default="viridis",
    )
    parser.add_argument("--zero-color", default="#151515")
    parser.add_argument("--outside-color", default="white")
    parser.add_argument("--graticule-color", default="#b8b8b8")
    parser.add_argument("--no-graticule", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--basename", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.repop_id < 0:
        raise ValueError("--repop-id must be non-negative.")
    if not hp.isnsideok(args.nside, nest=True):
        raise ValueError(f"Invalid NESTED NSIDE={args.nside}.")
    if args.xsize < 4 or args.xsize % 2:
        raise ValueError("--xsize must be an even integer of at least 4.")
    if not np.isfinite(args.panel_scale) or args.panel_scale <= 0.0:
        raise ValueError("--panel-scale must be positive and finite.")
    if args.dpi <= 0 or args.pdf_dpi <= 0 or args.chunk_size <= 0:
        raise ValueError("--dpi, --pdf-dpi, and --chunk-size must be positive.")
    if "pdf" in args.formats and args.pdf_dpi < 400:
        raise ValueError("PDF export requires --pdf-dpi >= 400.")
    if not (
        0.0
        <= args.lower_percentile
        < args.upper_percentile
        <= 100.0
    ):
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
    if args.basename is not None and Path(args.basename).name != args.basename:
        raise ValueError("--basename must be a filename stem, not a path.")
    for argument in ("zero_color", "outside_color", "graticule_color"):
        try:
            matplotlib.colors.to_rgba(getattr(args, argument))
        except ValueError as error:
            raise ValueError(f"Invalid --{argument.replace('_', '-')}.") from error


def output_paths(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], Path, Path]:
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("outputs")
        / "exploratory"
        / "auriga_clumpy_conservative_decomposition"
        / f"repop_{args.repop_id:04d}"
    )
    basename = args.basename or (
        f"auriga_clumpy_conservative_sum_repop_{args.repop_id:04d}_"
        f"nside{args.nside}_xsize{args.xsize}_{args.cmap}_{args.normalization}"
    )
    pngs = {
        scenario: output_dir / f"{basename}_{scenario}.png"
        for scenario in conventional.SCENARIOS
    }
    return (
        pngs,
        output_dir / f"{basename}.json",
        output_dir / f"{basename}_cell_counts.npz",
    )


def selected_figure_paths(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Path]], Path, Path]:
    png_paths, json_path, npz_path = output_paths(args)
    figure_paths = {
        scenario: {
            output_format: png_path.with_suffix(f".{output_format}")
            for output_format in args.formats
        }
        for scenario, png_path in png_paths.items()
    }
    return figure_paths, json_path, npz_path


def make_conservative_geometry(xsize: int) -> ConservativeGeometry:
    projector = hp.projector.MollweideProj(
        coord="G",
        xsize=xsize,
        flipconv="astro",
    )
    x, y = projector.ij2xy()
    inside = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
    valid_flat_cells = np.flatnonzero(inside.ravel()).astype(np.int64)
    valid_cell_xy = np.column_stack(
        (
            np.asarray(x)[inside],
            np.asarray(y)[inside],
        )
    )
    return ConservativeGeometry(
        projector=projector,
        xsize=xsize,
        ysize=xsize // 2,
        inside=inside,
        valid_flat_cells=valid_flat_cells,
        valid_cell_xy=valid_cell_xy,
        valid_cell_tree=cKDTree(valid_cell_xy),
    )


def assign_projected_coordinates(
    geometry: ConservativeGeometry,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Assign each finite projected point to exactly one valid raster cell."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    raw_i, raw_j = geometry.projector.xy2ij(x, y)
    i_mask = np.ma.getmaskarray(raw_i)
    j_mask = np.ma.getmaskarray(raw_j)
    i = np.ma.getdata(raw_i).astype(np.int64, copy=False)
    j = np.ma.getdata(raw_j).astype(np.int64, copy=False)
    finite = np.isfinite(x) & np.isfinite(y)
    in_bounds = (
        finite
        & ~i_mask
        & ~j_mask
        & (i >= 0)
        & (i < geometry.ysize)
        & (j >= 0)
        & (j < geometry.xsize)
    )
    direct = np.zeros(x.shape, dtype=np.bool_)
    direct[in_bounds] = geometry.inside[i[in_bounds], j[in_bounds]]

    flat_cells = np.full(x.shape, -1, dtype=np.int64)
    flat_cells[direct] = i[direct] * geometry.xsize + j[direct]

    remap = finite & ~direct
    max_remap_distance = 0.0
    if np.any(remap):
        distances, nearest = geometry.valid_cell_tree.query(
            np.column_stack((x[remap], y[remap])),
            k=1,
        )
        flat_cells[remap] = geometry.valid_flat_cells[nearest]
        max_remap_distance = float(np.max(distances))

    diagnostics: dict[str, int | float] = {
        "n_points": int(x.size),
        "n_direct_assignments": int(np.count_nonzero(direct)),
        "n_nearest_valid_cell_remaps": int(np.count_nonzero(remap)),
        "n_unassigned": int(np.count_nonzero(flat_cells < 0)),
        "max_nearest_remap_distance_projection_plane": max_remap_distance,
    }
    return flat_cells, diagnostics


def assign_healpix_pixels(
    geometry: ConservativeGeometry,
    nside: int,
    pixels: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | float]]:
    theta, phi = hp.pix2ang(nside, pixels, nest=True)
    x, y = geometry.projector.ang2xy(theta, phi, lonlat=False)
    return assign_projected_coordinates(geometry, x, y)


def increment_mapping_diagnostics(
    target: dict[str, int | float],
    update: dict[str, int | float],
) -> None:
    for key in (
        "n_points",
        "n_direct_assignments",
        "n_nearest_valid_cell_remaps",
        "n_unassigned",
    ):
        target[key] = int(target.get(key, 0)) + int(update[key])
    distance_key = "max_nearest_remap_distance_projection_plane"
    target[distance_key] = max(
        float(target.get(distance_key, 0.0)),
        float(update[distance_key]),
    )


def occupancy_histogram(counts: np.ndarray) -> dict[str, int]:
    unique, frequencies = np.unique(counts, return_counts=True)
    return {
        str(int(count)): int(frequency)
        for count, frequency in zip(unique, frequencies)
    }


def positive_occupancy_summary(counts: np.ndarray) -> dict[str, object]:
    occupied = counts[counts > 0]
    if not occupied.size:
        return {
            "n_cells": 0,
            "min": None,
            "max": None,
            "mean": None,
            "histogram_cells_by_contributor_count": {},
        }
    return {
        "n_cells": int(occupied.size),
        "min": int(occupied.min()),
        "max": int(occupied.max()),
        "mean": float(occupied.mean()),
        "histogram_cells_by_contributor_count": occupancy_histogram(occupied),
    }


def conservative_project(
    paths: dict[str, Path],
    metadata_by_scenario: dict[str, conventional.FitsMetadata],
    statistics: dict[str, dict[str, conventional.ComponentStatistics]],
    geometry: ConservativeGeometry,
    chunk_size: int,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, object],
]:
    n_cells = geometry.xsize * geometry.ysize
    linear_sums: dict[str, dict[str, np.ndarray]] = {}
    contributor_counts: dict[str, dict[str, np.ndarray]] = {}
    assignment_counts: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}

    for scenario in conventional.SCENARIOS:
        path = paths[scenario]
        metadata = metadata_by_scenario[scenario]
        scenario_sums = {
            component.key: np.zeros(n_cells, dtype=np.float64)
            for component in conventional.COMPONENTS
        }
        scenario_contributors = {
            component.key: np.zeros(n_cells, dtype=np.uint32)
            for component in conventional.COMPONENTS
        }
        scenario_assignments = np.zeros(n_cells, dtype=np.uint32)
        original_sums = {
            component.key: np.longdouble(0.0)
            for component in conventional.COMPONENTS
        }
        considered_positive = {component.key: 0 for component in conventional.COMPONENTS}
        mapping_diagnostics: dict[str, int | float] = {}

        print(f"Conservative projection: {scenario}: {path}", flush=True)
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            table = hdul[conventional.HDU_NAME].data
            for start in range(0, metadata.npix, chunk_size):
                stop = min(start + chunk_size, metadata.npix)
                pixels = np.asarray(
                    table[conventional.PIXEL_COLUMN][start:stop],
                    dtype=np.int64,
                )
                if not np.array_equal(pixels, np.arange(start, stop, dtype=np.int64)):
                    raise ValueError(
                        f"{path}: PIXEL is not ordered in rows {start}:{stop}."
                    )
                cell_ids, mapping_update = assign_healpix_pixels(
                    geometry,
                    metadata.nside,
                    pixels,
                )
                increment_mapping_diagnostics(mapping_diagnostics, mapping_update)
                assigned = cell_ids >= 0
                scenario_assignments += np.bincount(
                    cell_ids[assigned],
                    minlength=n_cells,
                ).astype(np.uint32)

                for component in conventional.COMPONENTS:
                    values, empty_sentinel = conventional.checked_values(
                        table,
                        component,
                        slice(start, stop),
                        metadata.conversion_factors[component.key],
                        path,
                    )
                    positive, _, _ = conventional.value_masks(values, empty_sentinel)
                    original_sums[component.key] += np.sum(
                        values[positive],
                        dtype=np.longdouble,
                    )
                    included = positive & assigned
                    considered_positive[component.key] += int(
                        np.count_nonzero(included)
                    )
                    included_cells = cell_ids[included]
                    scenario_sums[component.key] += np.bincount(
                        included_cells,
                        weights=values[included],
                        minlength=n_cells,
                    )
                    scenario_contributors[component.key] += np.bincount(
                        included_cells,
                        minlength=n_cells,
                    ).astype(np.uint32)
                print(
                    f"  {scenario}: rows {start:,}:{stop:,} / {metadata.npix:,}",
                    flush=True,
                )

        mapping_diagnostics["n_valid_raster_cells"] = int(
            geometry.valid_flat_cells.size
        )
        mapping_diagnostics["n_cells_receiving_healpix_centers"] = int(
            np.count_nonzero(scenario_assignments)
        )
        mapping_diagnostics["assignments_per_valid_cell"] = {
            "min": int(scenario_assignments[geometry.valid_flat_cells].min()),
            "max": int(scenario_assignments[geometry.valid_flat_cells].max()),
            "mean": float(
                scenario_assignments[geometry.valid_flat_cells].mean()
            ),
            "histogram_cells_by_assignment_count": occupancy_histogram(
                scenario_assignments[geometry.valid_flat_cells]
            ),
        }

        component_diagnostics = {}
        for component in conventional.COMPONENTS:
            key = component.key
            original_count = statistics[scenario][key].n_positive
            projected_values = scenario_sums[key]
            projected_positive = projected_values > 0.0
            original_sum = float(original_sums[key])
            projected_sum = float(
                np.sum(projected_values, dtype=np.longdouble)
            )
            absolute_error = projected_sum - original_sum
            relative_error = (
                absolute_error / original_sum if original_sum != 0.0 else 0.0
            )
            component_diagnostics[key] = {
                "original_occupied_healpix_pixels": original_count,
                "considered_occupied_healpix_pixels": considered_positive[key],
                "lost_occupied_healpix_pixels": (
                    original_count - considered_positive[key]
                ),
                "duplicated_healpix_contributions": 0,
                "projected_emitting_cells": int(
                    np.count_nonzero(projected_positive)
                ),
                "contributors_per_emitting_cell": positive_occupancy_summary(
                    scenario_contributors[key]
                ),
                "original_integrated_sum_j": original_sum,
                "projected_integrated_sum_j": projected_sum,
                "sum_absolute_error_j": absolute_error,
                "sum_relative_error": relative_error,
                "original_min_positive_j": statistics[scenario][key].min_positive_j,
                "original_max_j": statistics[scenario][key].max_j,
                "projected_min_positive_j": (
                    float(projected_values[projected_positive].min())
                    if np.any(projected_positive)
                    else None
                ),
                "projected_max_j": (
                    float(projected_values[projected_positive].max())
                    if np.any(projected_positive)
                    else None
                ),
            }

        component_sum = (
            scenario_sums["smooth"]
            + scenario_sums["pointlike"]
            + scenario_sums["extended"]
        )
        residual = scenario_sums["total"] - component_sum
        scale = np.maximum(np.abs(scenario_sums["total"]), np.abs(component_sum))
        nonclose = ~np.isclose(
            scenario_sums["total"],
            component_sum,
            rtol=1.0e-12,
            atol=0.0,
        )
        closure = {
            "sum_total_j": float(
                np.sum(scenario_sums["total"], dtype=np.longdouble)
            ),
            "sum_smooth_pointlike_extended_j": float(
                np.sum(component_sum, dtype=np.longdouble)
            ),
            "global_residual_j": float(
                np.sum(residual, dtype=np.longdouble)
            ),
            "l1_residual_j": float(
                np.sum(np.abs(residual), dtype=np.longdouble)
            ),
            "max_absolute_cell_residual_j": float(np.max(np.abs(residual))),
            "max_relative_cell_residual_where_defined": float(
                np.max(np.abs(residual[scale > 0.0]) / scale[scale > 0.0])
                if np.any(scale > 0.0)
                else 0.0
            ),
            "n_cells_not_close_rtol_1e-12": int(np.count_nonzero(nonclose)),
        }

        linear_sums[scenario] = scenario_sums
        contributor_counts[scenario] = scenario_contributors
        assignment_counts[scenario] = scenario_assignments
        diagnostics[scenario] = {
            "mapping": mapping_diagnostics,
            "components": component_diagnostics,
            "component_closure": closure,
        }

    return linear_sums, contributor_counts, assignment_counts, diagnostics


def conventional_sampling_diagnostics(
    paths: dict[str, Path],
    metadata_by_scenario: dict[str, conventional.FitsMetadata],
    statistics: dict[str, dict[str, conventional.ComponentStatistics]],
    xsize: int,
) -> dict[str, object]:
    geometry = conventional.make_projection_geometry(
        metadata_by_scenario[conventional.SCENARIOS[0]].nside,
        xsize,
    )
    result: dict[str, object] = {}
    for scenario in conventional.SCENARIOS:
        path = paths[scenario]
        metadata = metadata_by_scenario[scenario]
        components = {}
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            table = hdul[conventional.HDU_NAME].data
            for component in conventional.COMPONENTS:
                values, empty_sentinel = conventional.checked_values(
                    table,
                    component,
                    geometry.healpix_pixels,
                    metadata.conversion_factors[component.key],
                    path,
                )
                positive, _, _ = conventional.value_masks(values, empty_sentinel)
                sampled_pixels = geometry.healpix_pixels[positive]
                unique_sampled = np.unique(sampled_pixels)
                positive_values = values[positive]
                original_occupied = statistics[scenario][component.key].n_positive
                components[component.key] = {
                    "positive_raster_cells": int(positive_values.size),
                    "unique_occupied_healpix_pixels_sampled": int(
                        unique_sampled.size
                    ),
                    "occupied_healpix_pixels_not_sampled": int(
                        original_occupied - unique_sampled.size
                    ),
                    "fraction_of_occupied_healpix_pixels_sampled": (
                        float(unique_sampled.size / original_occupied)
                        if original_occupied
                        else None
                    ),
                    "duplicate_positive_raster_samples": int(
                        positive_values.size - unique_sampled.size
                    ),
                    "sum_of_sampled_raster_values_j_not_an_integral": float(
                        np.sum(positive_values, dtype=np.longdouble)
                    ),
                    "sampled_min_positive_j": (
                        float(positive_values.min())
                        if positive_values.size
                        else None
                    ),
                    "sampled_max_j": (
                        float(positive_values.max())
                        if positive_values.size
                        else None
                    ),
                }
        result[scenario] = {
            "n_inside_raster_cells": int(geometry.healpix_pixels.size),
            "components": components,
        }
    return result


def make_log_rasters(
    linear_sums: dict[str, dict[str, np.ndarray]],
    geometry: ConservativeGeometry,
) -> dict[str, dict[str, np.ndarray]]:
    projected = {scenario: {} for scenario in conventional.SCENARIOS}
    inside_flat = geometry.inside.ravel()
    for scenario in conventional.SCENARIOS:
        for component in conventional.COMPONENTS:
            values = linear_sums[scenario][component.key]
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(
                    f"{scenario}/{component.key}: invalid aggregated raster values."
                )
            raster = np.full(values.shape, np.nan, dtype=np.float64)
            raster[inside_flat] = conventional.ZERO_SENTINEL
            positive = values > 0.0
            raster[positive] = np.log10(values[positive])
            projected[scenario][component.key] = raster.reshape(
                geometry.ysize,
                geometry.xsize,
            )
    return projected


def projected_limits_and_extrema(
    projected: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, str],
    dict[str, dict[str, int | float]],
]:
    limits = {}
    sources = {}
    extrema = {}
    for component in conventional.COMPONENTS:
        values_by_scenario = []
        for scenario in conventional.SCENARIOS:
            raster = projected[scenario][component.key]
            positive = np.isfinite(raster) & (raster != conventional.ZERO_SENTINEL)
            values_by_scenario.append(raster[positive])
        values = np.concatenate(values_by_scenario)
        if not values.size:
            raise ValueError(f"{component.key}: no positive projected cells.")
        real_min = float(values.min())
        real_max = float(values.max())
        extrema[component.key] = {
            "min_positive_j": float(10.0 ** real_min),
            "max_j": float(10.0 ** real_max),
            "min_log10_j": real_min,
            "max_log10_j": real_max,
            "n_positive_projected_cells": int(values.size),
        }
        if args.normalization == "full":
            vmin, vmax = real_min, real_max
            source = "joint_full_aggregated_projected_cell_range"
        else:
            vmin, vmax = np.percentile(
                values,
                [args.lower_percentile, args.upper_percentile],
            )
            vmin, vmax = float(vmin), float(vmax)
            source = "joint_exact_projected_cell_percentiles"
        if vmax <= vmin:
            raise ValueError(
                f"{component.key}: degenerate projected color range {vmin}."
            )
        limits[component.key] = (vmin, vmax)
        sources[component.key] = source
    return limits, sources, extrema


def projected_outside_counts(
    projected: dict[str, dict[str, np.ndarray]],
    limits: dict[str, tuple[float, float]],
) -> dict[str, dict[str, dict[str, int]]]:
    result = {}
    for scenario in conventional.SCENARIOS:
        result[scenario] = {}
        for component in conventional.COMPONENTS:
            raster = projected[scenario][component.key]
            positive = np.isfinite(raster) & (raster != conventional.ZERO_SENTINEL)
            values = raster[positive]
            vmin, vmax = limits[component.key]
            below = int(np.count_nonzero(values < vmin))
            above = int(np.count_nonzero(values > vmax))
            result[scenario][component.key] = {
                "n_valid_positive": int(values.size),
                "n_below_vmin": below,
                "n_above_vmax": above,
                "n_outside": below + above,
            }
    return result


def save_cell_counts(
    path: Path,
    geometry: ConservativeGeometry,
    contributor_counts: dict[str, dict[str, np.ndarray]],
    assignment_counts: dict[str, np.ndarray],
) -> None:
    arrays = {"inside_mollweide": geometry.inside}
    for scenario in conventional.SCENARIOS:
        arrays[f"{scenario}_all_healpix_assignments"] = assignment_counts[
            scenario
        ].reshape(geometry.ysize, geometry.xsize)
        for component in conventional.COMPONENTS:
            arrays[f"{scenario}_{component.key}_positive_contributors"] = (
                contributor_counts[scenario][component.key].reshape(
                    geometry.ysize,
                    geometry.xsize,
                )
            )
    np.savez_compressed(path, **arrays)


def combined_outside_counts(
    counts: dict[str, dict[str, dict[str, int]]],
) -> dict[str, dict[str, int]]:
    return {
        component.key: {
            count_key: sum(
                counts[scenario][component.key][count_key]
                for scenario in conventional.SCENARIOS
            )
            for count_key in (
                "n_valid_positive",
                "n_below_vmin",
                "n_above_vmax",
                "n_outside",
            )
        }
        for component in conventional.COMPONENTS
    }


def write_diagnostics(
    path: Path,
    npz_path: Path,
    input_paths: dict[str, Path],
    output_pngs: dict[str, Path],
    statistics: dict[str, dict[str, conventional.ComponentStatistics]],
    limits: dict[str, tuple[float, float]],
    limit_sources: dict[str, str],
    projected_extrema: dict[str, dict[str, int | float]],
    outside_counts: dict[str, dict[str, dict[str, int]]],
    conservative_diagnostics: dict[str, object],
    conventional_diagnostics: dict[str, object],
    canvas_sizes: dict[str, tuple[int, int]],
    args: argparse.Namespace,
    output_pdfs: dict[str, Path] | None = None,
) -> None:
    panel_width = int(round(args.xsize * args.panel_scale))
    panel_height = int(round((args.xsize // 2) * args.panel_scale))
    export_dpi = {
        "png": args.dpi,
        "pdf": args.pdf_dpi,
    }
    estimated_map_artist_pixels = {
        output_format: {
            "width": int(round(panel_width * export_dpi[output_format] / args.dpi)),
            "height": int(
                round(panel_height * export_dpi[output_format] / args.dpi)
            ),
        }
        for output_format in args.formats
    }
    payload = {
        "prototype": "experimental conservative-sum Mollweide reprojection",
        "inputs": {key: str(value) for key, value in input_paths.items()},
        "pngs": {key: str(value) for key, value in output_pngs.items()},
        "cell_count_npz": str(npz_path),
        "repop_id": args.repop_id,
        "nside": args.nside,
        "ordering": "NESTED",
        "coordinates": "Galactic",
        "quantity": (
            "integrated J-factor summed over HEALPix pixel centers assigned "
            "to each Mollweide raster cell"
        ),
        "method": {
            "assignment": (
                "HEALPix center -> healpy Mollweide ang2xy -> nearest raster "
                "cell; rounded cells outside the displayed ellipse are "
                "remapped to the nearest valid cell center"
            ),
            "aggregation": "sum",
            "one_assignment_per_healpix_pixel": True,
            "smoothing": False,
            "dilation": False,
            "markers": False,
        },
        "boundary_behavior": {
            "longitude_seam": (
                "Longitudes immediately below +180 and above -180 map to "
                "opposite ellipse edges; each HEALPix center is assigned "
                "once, so the seam can split morphology but cannot duplicate "
                "integrated J."
            ),
            "poles": (
                "Longitude convergence near the poles produces many-to-one "
                "cell assignments; these collisions are summed."
            ),
            "ellipse_edge": (
                "If healpy rounding selects a cell whose center is outside "
                "the displayed ellipse, the point is assigned to the nearest "
                "valid center and the count/displacement are recorded."
            ),
        },
        "projection": {
            "xsize": args.xsize,
            "ysize": args.xsize // 2,
            "panel_scale": args.panel_scale,
            "canvas_pixels": {
                scenario: {
                    "width": canvas_sizes[scenario][0],
                    "height": canvas_sizes[scenario][1],
                }
                for scenario in conventional.SCENARIOS
            },
        },
        "rendering": {
            "cmap": args.cmap,
            "zero_color": args.zero_color,
            "interpolation": "nearest",
            "formats": list(args.formats),
            "png_and_layout_dpi": args.dpi,
            "pdf_raster_dpi": args.pdf_dpi,
            "pdf_text_and_annotations_remain_vector": True,
            "projected_map_raster_pixels": {
                "width": args.xsize,
                "height": args.xsize // 2,
            },
            "estimated_map_artist_pixels_at_export_dpi": (
                estimated_map_artist_pixels
            ),
            "export_samples_per_projected_cell": {
                output_format: (
                    args.panel_scale * export_dpi[output_format] / args.dpi
                )
                for output_format in args.formats
            },
        },
        "scale_selection": {
            "mode": args.normalization,
            "domain": "aggregated_projected_cells",
            "transform": "log10",
            "shared_between_scenarios_within_each_component": True,
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "limits": {
                component.key: {
                    "vmin": limits[component.key][0],
                    "vmax": limits[component.key][1],
                    "source": limit_sources[component.key],
                }
                for component in conventional.COMPONENTS
            },
            "projected_real_extrema": projected_extrema,
            "valid_positive_outside_scale": {
                "by_scenario": outside_counts,
                "combined_by_component": combined_outside_counts(outside_counts),
            },
        },
        "original_healpix_statistics": conventional.serializable_statistics(
            statistics
        ),
        "conservative_sum": conservative_diagnostics,
        "conventional_center_sampling": conventional_diagnostics,
        "computational_strategy": {
            "full_map_passes_per_scenario": 2,
            "pass_1": "chunked original HEALPix statistics",
            "pass_2": "chunked projection and aggregation",
            "conventional_comparison": (
                "indexed read of only the conventional raster samples"
            ),
            "chunk_size": args.chunk_size,
            "both_scenarios_are_never_open_simultaneously": True,
            "estimated_persistent_raster_bytes": (
                168 * args.xsize * (args.xsize // 2)
            ),
        },
        "interpretation": {
            "sum_conservation": (
                "Conservative-sum preserves the full-sky integrated sum up "
                "to floating-point accumulation error."
            ),
            "not_intensity_preserving": (
                "Projected values are sums over a variable number of equal-"
                "area HEALPix pixels; they are not surface brightness unless "
                "divided by an explicitly chosen projected solid angle."
            ),
            "effective_resolution": (
                "The raster resolution and variable assignments per cell set "
                "the displayed angular resolution; it is not the NSIDE=2048 "
                "HEALPix resolution."
            ),
        },
    }
    if output_pdfs:
        payload["pdfs"] = {
            key: str(value) for key, value in output_pdfs.items()
        }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    input_paths = conventional.resolve_map_paths(args)
    figure_paths, json_path, npz_path = selected_figure_paths(args)
    figure_paths = {
        scenario: {
            output_format: path.expanduser().resolve()
            for output_format, path in paths.items()
        }
        for scenario, paths in figure_paths.items()
    }
    json_path = json_path.expanduser().resolve()
    npz_path = npz_path.expanduser().resolve()
    existing = [
        path
        for path in (
            *(
                path
                for paths in figure_paths.values()
                for path in paths.values()
            ),
            json_path,
            npz_path,
        )
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )

    statistics, metadata = conventional.scan_value_ranges(
        input_paths,
        args.nside,
        args.chunk_size,
    )
    geometry = make_conservative_geometry(args.xsize)
    (
        linear_sums,
        contributor_counts,
        assignment_counts,
        conservative_diagnostics,
    ) = conservative_project(
        input_paths,
        metadata,
        statistics,
        geometry,
        args.chunk_size,
    )
    projected = make_log_rasters(linear_sums, geometry)
    limits, limit_sources, projected_extrema = projected_limits_and_extrema(
        projected,
        args,
    )
    outside_counts = projected_outside_counts(projected, limits)
    if args.normalization == "full":
        outside = combined_outside_counts(outside_counts)
        if any(values["n_outside"] for values in outside.values()):
            raise RuntimeError(
                "Full normalization left aggregated projected cells outside "
                "the color scale."
            )
    conventional_diagnostics = conventional_sampling_diagnostics(
        input_paths,
        metadata,
        statistics,
        args.xsize,
    )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_cell_counts(
        npz_path,
        geometry,
        contributor_counts,
        assignment_counts,
    )
    canvas_sizes = {}
    for scenario in conventional.SCENARIOS:
        for output_format in args.formats:
            canvas_size = conventional.render_figure(
                scenario,
                projected,
                limits,
                figure_paths[scenario][output_format],
                args,
            )
            if scenario in canvas_sizes and canvas_sizes[scenario] != canvas_size:
                raise RuntimeError(
                    f"Inconsistent canvas size between output formats for {scenario}."
                )
            canvas_sizes[scenario] = canvas_size
    png_paths = {
        scenario: paths["png"]
        for scenario, paths in figure_paths.items()
        if "png" in paths
    }
    pdf_paths = {
        scenario: paths["pdf"]
        for scenario, paths in figure_paths.items()
        if "pdf" in paths
    }
    write_diagnostics(
        json_path,
        npz_path,
        input_paths,
        png_paths,
        statistics,
        limits,
        limit_sources,
        projected_extrema,
        outside_counts,
        conservative_diagnostics,
        conventional_diagnostics,
        canvas_sizes,
        args,
        output_pdfs=pdf_paths,
    )

    for scenario in conventional.SCENARIOS:
        for output_format in args.formats:
            print(
                f"{output_format.upper()} ({scenario}): "
                f"{figure_paths[scenario][output_format]}"
            )
    print(f"JSON: {json_path}")
    print(f"Cell counts NPZ: {npz_path}")


if __name__ == "__main__":
    main()
