#!/usr/bin/env python3
"""Compare quantitative subhalo and total Auriga--CLUMPY sky regions.

The displayed scientific quantity is constructed directly from the integrated
J-factor columns of each final map,

    J_sub = Jpointlike + Jextended,
    J_tot = Jsmooth + Jpointlike + Jextended.

Each value remains attached to its native NESTED HEALPix pixel. The pixel
boundaries are projected to a local gnomonic plane and drawn as flat-valued
polygons. There is no raster resampling, interpolation, smoothing, dilation,
component overlay, or display boost. Positive values use one logarithmic
normalization per column, shared between the fragile and resilient scenarios,
and each column has its own quantitative colorbar. Exact zeros, which are
outside the domain of a logarithmic normalization, are retained and drawn as
opaque black polygons.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import healpy as hp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import plot_auriga_clumpy_conservative_decomposition as fullsky


SCENARIOS = fullsky.SCENARIOS
DEFAULT_NSIDE = 2048
DEFAULT_ROI_WIDTH_DEG = 20.0
DEFAULT_ROI_HEIGHT_DEG = 10.0
DEFAULT_DPI = 500
DEFAULT_CMAP = "viridis"
ZERO_COLOR = "#000000"
SUPPORTED_FORMATS = ("png", "pdf")
ROI_CENTERS = {
    "roi_01": (10.0, -30.0),
    "roi_02": (355.0, 0.0),
    "roi_03": (30.0, -5.0),
    "roi_04": (340.0, 5.0),
}
# Compensate for scaling the 13-inch PDF to a 440 TeX pt text width.
PANEL_TITLE_FONTSIZE = 22.3
LABEL_FONTSIZE = 20.2
TICK_FONTSIZE = 18.1
LEGEND_FONTSIZE = 18.1


@dataclass(frozen=True)
class TangentFrame:
    longitude_deg: float
    latitude_deg: float
    center: np.ndarray
    east: np.ndarray
    north: np.ndarray
    half_width_tan: float
    half_height_tan: float


@dataclass(frozen=True)
class RoiGeometry:
    roi_id: str
    frame: TangentFrame
    intersecting_pixels: np.ndarray
    center_inside: np.ndarray
    n_roundtrip_mismatches: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare J_sub and J_tot in fragile and resilient ROIs using "
            "native HEALPix polygons."
        )
    )
    parser.add_argument("--repop-id", type=int, default=0)
    parser.add_argument("--nside", type=int, default=DEFAULT_NSIDE)
    parser.add_argument("--maps-root", type=Path, default=Path("outputs/clumpy"))
    parser.add_argument("--fragile-map", type=Path, default=None)
    parser.add_argument("--resilient-map", type=Path, default=None)
    parser.add_argument(
        "--fragile-roi",
        choices=tuple(ROI_CENTERS),
        default="roi_04",
        help="ROI used in the fragile panel (default: roi_04).",
    )
    parser.add_argument(
        "--resilient-roi",
        choices=tuple(ROI_CENTERS),
        default="roi_04",
        help="ROI used in the resilient panel (default: roi_04).",
    )
    parser.add_argument("--roi-width-deg", type=float, default=DEFAULT_ROI_WIDTH_DEG)
    parser.add_argument("--roi-height-deg", type=float, default=DEFAULT_ROI_HEIGHT_DEG)
    parser.add_argument("--cmap", default=DEFAULT_CMAP)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=SUPPORTED_FORMATS,
        default=SUPPORTED_FORMATS,
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--basename", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.repop_id < 0:
        raise ValueError("--repop-id must be non-negative.")
    if not hp.isnsideok(args.nside, nest=True):
        raise ValueError(f"Invalid input NSIDE={args.nside}.")
    if not (0.0 < args.roi_width_deg < 90.0):
        raise ValueError("--roi-width-deg must lie between 0 and 90 degrees.")
    if not (0.0 < args.roi_height_deg < 90.0):
        raise ValueError("--roi-height-deg must lie between 0 and 90 degrees.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if "pdf" in args.formats and args.dpi < 400:
        raise ValueError("PDF export requires --dpi >= 400.")
    if args.cmap not in matplotlib.colormaps:
        raise ValueError(f"Unknown Matplotlib colormap: {args.cmap!r}.")
    if args.basename is not None:
        basename = str(args.basename).strip()
        if not basename or Path(basename).name != basename:
            raise ValueError("--basename must be a non-empty filename stem.")


def resolve_map_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {}
    for scenario in SCENARIOS:
        override = getattr(args, f"{scenario}_map")
        path = (
            Path(override)
            if override is not None
            else fullsky.expected_map_path(
                args.maps_root,
                scenario,
                args.repop_id,
                args.nside,
            )
        )
        path = path.expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{scenario}: FITS not found or empty: {path}")
        paths[scenario] = path
    return paths


def output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    return (
        Path("outputs")
        / "diagnostics"
        / "maps"
        / "subhalo_rois"
        / f"repop_{args.repop_id:04d}"
    ).resolve()


def output_basename(args: argparse.Namespace) -> str:
    if args.basename is not None:
        return str(args.basename).strip()
    return (
        f"subhalo_total_fragile_{args.fragile_roi}_resilient_{args.resilient_roi}"
        f"_nside{args.nside}"
    )


def tangent_frame(
    longitude_deg: float,
    latitude_deg: float,
    width_deg: float,
    height_deg: float,
) -> TangentFrame:
    longitude = math.radians(longitude_deg)
    latitude = math.radians(latitude_deg)
    center = np.array(
        [
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        ]
    )
    east = np.array([-math.sin(longitude), math.cos(longitude), 0.0])
    north = np.array(
        [
            -math.sin(latitude) * math.cos(longitude),
            -math.sin(latitude) * math.sin(longitude),
            math.cos(latitude),
        ]
    )
    return TangentFrame(
        longitude_deg=longitude_deg,
        latitude_deg=latitude_deg,
        center=center,
        east=east,
        north=north,
        half_width_tan=math.tan(math.radians(width_deg / 2.0)),
        half_height_tan=math.tan(math.radians(height_deg / 2.0)),
    )


def project_vectors(
    vectors: np.ndarray,
    frame: TangentFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denominator = vectors @ frame.center
    x = -(vectors @ frame.east) / denominator
    y = (vectors @ frame.north) / denominator
    return x, y, denominator


def inverse_projected_vector(x: float, y: float, frame: TangentFrame) -> np.ndarray:
    vector = frame.center - x * frame.east + y * frame.north
    return vector / np.linalg.norm(vector)


def roi_corner_vectors(frame: TangentFrame) -> np.ndarray:
    return np.asarray(
        [
            inverse_projected_vector(x, y, frame)
            for x, y in (
                (-frame.half_width_tan, -frame.half_height_tan),
                (frame.half_width_tan, -frame.half_height_tan),
                (frame.half_width_tan, frame.half_height_tan),
                (-frame.half_width_tan, frame.half_height_tan),
            )
        ]
    )


def make_roi_geometry(
    nside: int,
    roi_id: str,
    width_deg: float,
    height_deg: float,
) -> RoiGeometry:
    longitude_deg, latitude_deg = ROI_CENTERS[roi_id]
    frame = tangent_frame(longitude_deg, latitude_deg, width_deg, height_deg)
    pixels = hp.query_polygon(
        nside,
        roi_corner_vectors(frame),
        inclusive=True,
        fact=4,
        nest=True,
    ).astype(np.int64)
    vectors = np.column_stack(hp.pix2vec(nside, pixels, nest=True))
    x, y, denominator = project_vectors(vectors, frame)
    if np.any(denominator <= 0.0):
        raise ValueError(f"{roi_id}: selected a pixel beyond the gnomonic horizon.")
    center_inside = (np.abs(x) <= frame.half_width_tan) & (
        np.abs(y) <= frame.half_height_tan
    )
    recovered = hp.vec2pix(
        nside,
        vectors[:, 0],
        vectors[:, 1],
        vectors[:, 2],
        nest=True,
    )
    return RoiGeometry(
        roi_id=roi_id,
        frame=frame,
        intersecting_pixels=pixels,
        center_inside=center_inside,
        n_roundtrip_mismatches=int(np.count_nonzero(recovered != pixels)),
    )


def read_roi_values(
    path: Path,
    expected_nside: int,
    pixels: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Read native components and construct the two displayed quantities."""
    component_specs = {
        component.key: component
        for component in fullsky.COMPONENTS
        if component.key in {"smooth", "pointlike", "extended"}
    }
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        if fullsky.HDU_NAME not in hdul:
            raise ValueError(f"{path}: missing {fullsky.HDU_NAME!r} HDU.")
        hdu = hdul[fullsky.HDU_NAME]
        metadata = fullsky.validate_hdu(hdu, path, expected_nside)
        table = hdu.data
        if table is None:
            raise ValueError(f"{path} [{fullsky.HDU_NAME}]: missing table data.")
        stored_pixels = np.asarray(
            table[fullsky.PIXEL_COLUMN][pixels],
            dtype=np.int64,
        )
        if not np.array_equal(stored_pixels, pixels):
            raise ValueError(f"{path}: indexed PIXEL values do not match row indices.")

        components = {}
        sentinel_counts = {}
        for key in ("smooth", "pointlike", "extended"):
            specification = component_specs[key]
            values, empty_sentinel = fullsky.checked_values(
                table,
                specification,
                pixels,
                metadata.conversion_factors[key],
                path,
            )
            _, zero, invalid = fullsky.value_masks(values, empty_sentinel)
            if np.any(invalid):
                raise ValueError(
                    f"{path}:{specification.column}: found "
                    f"{np.count_nonzero(invalid)} invalid ROI values."
                )
            values = values.copy()
            values[zero] = 0.0
            components[key] = values
            sentinel_counts[key] = int(np.count_nonzero(empty_sentinel))

    subhalo = components["pointlike"] + components["extended"]
    total = components["smooth"] + subhalo
    fields = {"J_sub": subhalo, "J_tot": total}
    for field_name, values in fields.items():
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{path}: invalid constructed {field_name} values.")
    diagnostics = {
        "source_hdu": fullsky.HDU_NAME,
        "source_columns": ["Jsmooth", "Jpointlike", "Jextended"],
        "field_source_columns": {
            "J_sub": ["Jpointlike", "Jextended"],
            "J_tot": ["Jsmooth", "Jpointlike", "Jextended"],
        },
        "equations": {
            "J_sub": "Jpointlike + Jextended",
            "J_tot": "Jsmooth + Jpointlike + Jextended",
        },
        "unit": str(fullsky.EXPECTED_UNIT),
        "n_indexed_pixels": int(pixels.size),
        "pixel_counts": {
            field_name: {
                "positive": int(np.count_nonzero(values > 0.0)),
                "zero": int(np.count_nonzero(values == 0.0)),
            }
            for field_name, values in fields.items()
        },
        "n_empty_extended_sentinel_converted_to_zero": sentinel_counts["extended"],
        "component_sums": {
            "Jsmooth": float(np.sum(components["smooth"], dtype=np.longdouble)),
            "Jpointlike": float(np.sum(components["pointlike"], dtype=np.longdouble)),
            "Jextended": float(np.sum(components["extended"], dtype=np.longdouble)),
            "J_sub": float(np.sum(subhalo, dtype=np.longdouble)),
            "J_tot": float(np.sum(total, dtype=np.longdouble)),
        },
        "read_Jsmooth": True,
        "read_Jtot_for_display": False,
    }
    return fields, diagnostics


def read_subhalo_values(
    path: Path,
    expected_nside: int,
    pixels: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Backward-compatible subhalo reader used by existing diagnostics/tests."""
    fields, diagnostics = read_roi_values(path, expected_nside, pixels)
    subhalo_diagnostics = {
        **diagnostics,
        "equation": "J_sub = Jpointlike + Jextended",
        "n_positive_pixels": diagnostics["pixel_counts"]["J_sub"]["positive"],
        "n_zero_pixels": diagnostics["pixel_counts"]["J_sub"]["zero"],
    }
    return fields["J_sub"], subhalo_diagnostics


def projected_pixel_polygons(
    nside: int,
    pixels: np.ndarray,
    frame: TangentFrame,
) -> np.ndarray:
    boundaries = hp.boundaries(nside, pixels, step=1, nest=True)
    vectors = np.transpose(boundaries, (0, 2, 1))
    denominator = vectors @ frame.center
    if np.any(denominator <= 0.0):
        raise ValueError("ROI pixel boundary crossed the gnomonic horizon.")
    x = -(vectors @ frame.east) / denominator
    y = (vectors @ frame.north) / denominator
    return np.stack((x, y), axis=-1)


def positive_distribution(values: np.ndarray) -> dict[str, float | int]:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return {"count": 0}
    return {
        "count": int(positive.size),
        "min": float(positive.min()),
        "max": float(positive.max()),
    }


def shared_log_norm(
    values_by_scenario: dict[str, np.ndarray],
) -> tuple[LogNorm, dict[str, object]]:
    """Build an exact positive-domain norm shared by both scenarios."""
    named_values = {}
    for scenario in SCENARIOS:
        named_values[scenario] = np.asarray(
            values_by_scenario[scenario], dtype=np.float64
        )
    positive_arrays = {
        name: values[values > 0.0] for name, values in named_values.items()
    }
    nonempty = [values for values in positive_arrays.values() if values.size]
    if not nonempty:
        raise ValueError("No positive pixels in either scenario.")
    joint = np.concatenate(nonempty)
    exact_vmin = float(joint.min())
    exact_vmax = float(joint.max())
    if exact_vmax <= exact_vmin:
        raise ValueError(
            "Positive pixels have a constant value; an exact LogNorm range "
            "requires distinct minimum and maximum values."
        )
    norm = LogNorm(vmin=exact_vmin, vmax=exact_vmax, clip=False)
    diagnostics = {
        "type": "matplotlib.colors.LogNorm",
        "shared_between_scenarios": True,
        "input_values": "native integrated J per HEALPix pixel",
        "positive_domain_only": True,
        "vmin": exact_vmin,
        "vmax": exact_vmax,
        "exact_joint_positive_min": exact_vmin,
        "exact_joint_positive_max": exact_vmax,
        "expanded_constant_range": False,
        "percentile_clipping": False,
        "artificial_floor": False,
        "distributions": {
            name: positive_distribution(values)
            for name, values in named_values.items()
        }
        | {"joint": positive_distribution(joint)},
    }
    return norm, diagnostics


def column_log_norms(
    values_by_scenario: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, LogNorm], dict[str, object]]:
    """Build independent exact norms for J_sub and J_tot columns."""
    norms = {}
    columns = {}
    for field_name in ("J_sub", "J_tot"):
        norm, diagnostics = shared_log_norm(
            {
                scenario: values_by_scenario[scenario][field_name]
                for scenario in SCENARIOS
            }
        )
        norms[field_name] = norm
        columns[field_name] = diagnostics
    return norms, {
        "type": "matplotlib.colors.LogNorm",
        "strategy": "independent exact positive range per column",
        "shared_between_scenarios_within_each_column": True,
        "shared_between_columns": False,
        "percentile_clipping": False,
        "artificial_floor": False,
        "columns": columns,
    }


def configure_offset_ticks(
    axes: plt.Axes,
    frame: TangentFrame,
    width_deg: float,
    height_deg: float,
) -> None:
    """Label central-axis crossings in Galactic angular offsets.

    A constant-longitude locus is generally curved in a gnomonic panel.  Its
    intersection with the horizontal central axis (y=0) is
    x = -cos(b0) tan(delta_longitude).  Along the vertical central axis (x=0),
    y = tan(delta_latitude).  The minus sign retains the astronomical
    convention of increasing Galactic longitude toward the left.
    """
    longitude_offsets = np.linspace(width_deg / 2.0, -width_deg / 2.0, 5)
    latitude_offsets = np.linspace(-height_deg / 2.0, height_deg / 2.0, 5)
    central_latitude = math.radians(frame.latitude_deg)
    axes.set_xticks(
        -math.cos(central_latitude) * np.tan(np.radians(longitude_offsets))
    )
    axes.set_xticklabels([f"{value:g}" for value in longitude_offsets])
    axes.set_yticks(np.tan(np.radians(latitude_offsets)))
    axes.set_yticklabels([f"{value:g}" for value in latitude_offsets])


def scientific_tick_label(value: float, _position: int | None = None) -> str:
    """Format a positive colorbar tick compactly without an axis offset."""
    if not np.isfinite(value) or value <= 0.0:
        return ""
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / 10.0**exponent
    return rf"${mantissa:.3g}\times10^{{{exponent}}}$"


def plot_style() -> dict[str, object]:
    """Match the typographic hierarchy of the official all-sky figures."""
    return {
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


def render_comparison(
    output_paths: dict[str, Path],
    geometries: dict[str, RoiGeometry],
    values_by_scenario: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, object]:
    with plt.rc_context(plot_style()):
        return _render_comparison(
            output_paths,
            geometries,
            values_by_scenario,
            args,
        )


def _render_comparison(
    output_paths: dict[str, Path],
    geometries: dict[str, RoiGeometry],
    values_by_scenario: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, object]:
    norms, normalization_diagnostics = column_log_norms(values_by_scenario)
    cmap = matplotlib.colormaps[args.cmap].copy()
    cmap.set_bad(ZERO_COLOR, alpha=1.0)
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.2), squeeze=False)
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.24,
        top=0.93,
        wspace=0.10,
        hspace=0.36,
    )
    rendered_counts = {}

    for row, scenario in enumerate(SCENARIOS):
        geometry = geometries[scenario]
        polygons = projected_pixel_polygons(
            args.nside,
            geometry.intersecting_pixels,
            geometry.frame,
        )
        rendered_counts[scenario] = {}
        for column, field_name in enumerate(("J_sub", "J_tot")):
            axes_item = axes[row, column]
            values = values_by_scenario[scenario][field_name]
            plotted_values = np.ma.masked_equal(values, 0.0, copy=False)
            collection = PolyCollection(
                polygons,
                array=plotted_values,
                cmap=cmap,
                norm=norms[field_name],
                edgecolors="none",
                linewidths=0.0,
                antialiased=False,
                rasterized=True,
            )
            axes_item.add_collection(collection)
            axes_item.set_facecolor(ZERO_COLOR)
            axes_item.set_xlim(
                -geometry.frame.half_width_tan,
                geometry.frame.half_width_tan,
            )
            axes_item.set_ylim(
                -geometry.frame.half_height_tan,
                geometry.frame.half_height_tan,
            )
            axes_item.set_aspect("equal", adjustable="box")
            configure_offset_ticks(
                axes_item,
                geometry.frame,
                args.roi_width_deg,
                args.roi_height_deg,
            )
            if row == 1:
                axes_item.set_xlabel(
                    r"Galactic longitude offset $\Delta\ell$ [deg]"
                )
            if column == 0:
                axes_item.set_ylabel(r"Galactic latitude offset $\Delta b$ [deg]")
            axes_item.set_title(
                scenario.capitalize(),
                pad=9.0,
                fontweight="bold",
            )
            rendered_counts[scenario][field_name] = int(values.size)

    colorbar_axes = []
    colorbar_diagnostics = []
    for column, field_name in enumerate(("J_sub", "J_tot")):
        panel_position = axes[1, column].get_position()
        inset = 0.035
        colorbar_axis = figure.add_axes(
            (
                panel_position.x0 + inset,
                0.105,
                panel_position.width - 2.0 * inset,
                0.026,
            )
        )
        scalar_mappable = ScalarMappable(norm=norms[field_name], cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = figure.colorbar(
            scalar_mappable,
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar_ticks = np.geomspace(
            float(norms[field_name].vmin),
            float(norms[field_name].vmax),
            num=4,
        )
        colorbar.set_ticks(colorbar_ticks)
        colorbar.ax.xaxis.set_major_formatter(FuncFormatter(scientific_tick_label))
        colorbar.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)
        symbol = (
            r"J_{\mathrm{sub}}" if field_name == "J_sub" else r"J_{\mathrm{tot}}"
        )
        colorbar.set_label(
            rf"${symbol}\;[\mathrm{{GeV^2\,cm^{{-5}}}}]$",
            labelpad=5.0,
        )
        colorbar_axes.append(colorbar_axis)
        colorbar_diagnostics.append(
            {
                "column": field_name,
                "vmin": float(norms[field_name].vmin),
                "vmax": float(norms[field_name].vmax),
                "ticks": [float(value) for value in colorbar_ticks],
                "unit": "GeV2 cm-5 per native HEALPix pixel",
            }
        )
    zero_indicator = None
    if any(
        np.any(values_by_scenario[scenario][field_name] == 0.0)
        for scenario in SCENARIOS
        for field_name in ("J_sub", "J_tot")
    ):
        # Anchor the separate zero indicator beyond the full visual extent of
        # the J_sub colorbar, including its leftmost tick label.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        j_sub_colorbar_bound = colorbar_axes[0].get_tightbbox(renderer)
        j_sub_colorbar_position = colorbar_axes[0].get_position()
        indicator_gap = 0.008
        zero_indicator = figure.legend(
            handles=[
                Patch(
                    facecolor=ZERO_COLOR,
                    edgecolor="0.35",
                    label=r"$J=0$",
                )
            ],
            loc="center right",
            bbox_to_anchor=(
                j_sub_colorbar_bound.x0 / figure.bbox.width - indicator_gap,
                j_sub_colorbar_position.y0
                + j_sub_colorbar_position.height / 2.0,
            ),
            bbox_transform=figure.transFigure,
            frameon=False,
            borderaxespad=0.0,
            handlelength=0.5,
            handleheight=0.6,
            handletextpad=0.3,
        )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panel_bounds = [axes_item.get_tightbbox(renderer) for axes_item in axes.flat]
    colorbar_bounds = [axis.get_tightbbox(renderer) for axis in colorbar_axes]
    panels_overlap_each_other = bool(
        any(
            left.overlaps(right)
            for index, left in enumerate(panel_bounds)
            for right in panel_bounds[index + 1 :]
        )
    )
    colorbars_overlap_panels = bool(
        any(
            colorbar_bound.overlaps(panel_bound)
            for colorbar_bound in colorbar_bounds
            for panel_bound in panel_bounds
        )
    )
    colorbars_overlap_each_other = bool(
        colorbar_bounds[0].overlaps(colorbar_bounds[1])
    )
    zero_indicator_bound = (
        zero_indicator.get_window_extent(renderer)
        if zero_indicator is not None
        else None
    )
    figure_bound = figure.bbox
    zero_indicator_within_figure = bool(
        zero_indicator_bound is None
        or (
            zero_indicator_bound.x0 >= figure_bound.x0
            and zero_indicator_bound.y0 >= figure_bound.y0
            and zero_indicator_bound.x1 <= figure_bound.x1
            and zero_indicator_bound.y1 <= figure_bound.y1
        )
    )
    zero_indicator_overlaps_colorbars = bool(
        zero_indicator_bound is not None
        and any(
            zero_indicator_bound.overlaps(colorbar_bound)
            for colorbar_bound in colorbar_bounds
        )
    )
    zero_indicator_overlaps_panels = bool(
        zero_indicator_bound is not None
        and any(
            zero_indicator_bound.overlaps(panel_bound)
            for panel_bound in panel_bounds
        )
    )
    colorbar_tick_labels_overlap = False
    for colorbar_axis in colorbar_axes:
        tick_bounds = [
            label.get_window_extent(renderer)
            for label in colorbar_axis.get_xticklabels()
            if label.get_visible() and label.get_text()
        ]
        colorbar_tick_labels_overlap = colorbar_tick_labels_overlap or any(
            left.overlaps(right)
            for left, right in zip(tick_bounds, tick_bounds[1:])
        )
    for output_path in output_paths.values():
        figure.savefig(output_path, dpi=args.dpi, facecolor="white")
    figure_size = {
        "width": float(figure.get_figwidth()),
        "height": float(figure.get_figheight()),
    }
    plt.close(figure)

    return {
        "paths": {key: str(value) for key, value in output_paths.items()},
        "colormap": args.cmap,
        "normalization": normalization_diagnostics,
        "quantitative_colorbars": {
            "count": 2,
            "orientation": "horizontal",
            "one_below_each_column": True,
            "units": "GeV2 cm-5 per native HEALPix pixel",
            "columns": colorbar_diagnostics,
        },
        "zero_value_rendering": {
            "color": ZERO_COLOR,
            "opaque": True,
            "reason": "zero is outside the domain of LogNorm",
            "positive_floor_inserted": False,
        },
        "rendered_native_pixels": rendered_counts,
        "all_intersecting_pixels_rendered": all(
            rendered_counts[scenario][field_name]
            == geometries[scenario].intersecting_pixels.size
            for scenario in SCENARIOS
            for field_name in ("J_sub", "J_tot")
        ),
        "panel_layout": {
            "shape": [2, 2],
            "rows": ["fragile", "resilient"],
            "columns": ["J_sub", "J_tot"],
            "same_roi_within_each_scenario_row": True,
        },
        "layout_validation": {
            "suptitle_present": getattr(figure, "_suptitle", None) is not None,
            "panel_titles_single_line": all(
                "\n" not in axes_item.get_title() for axes_item in axes.flat
            ),
            "panels_overlap_each_other": panels_overlap_each_other,
            "colorbars_overlap_panels": colorbars_overlap_panels,
            "colorbars_overlap_each_other": colorbars_overlap_each_other,
            "colorbar_tick_labels_overlap": bool(colorbar_tick_labels_overlap),
            "zero_indicator_present": zero_indicator is not None,
            "zero_indicator_within_figure": zero_indicator_within_figure,
            "zero_indicator_overlaps_colorbars": (
                zero_indicator_overlaps_colorbars
            ),
            "zero_indicator_overlaps_panels": zero_indicator_overlaps_panels,
        },
        "figure_size_inches": figure_size,
        "rasterized_artist_dpi": args.dpi,
        "native_polygon_collections_rasterized_at_export": True,
        "text_axes_and_colorbar_labels_remain_vector_in_pdf": True,
    }


def geometry_diagnostics(
    geometry: RoiGeometry,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "roi_id": geometry.roi_id,
        "center_longitude_deg": geometry.frame.longitude_deg,
        "center_latitude_deg": geometry.frame.latitude_deg,
        "width_deg": args.roi_width_deg,
        "height_deg": args.roi_height_deg,
        "n_pixels_intersecting_roi": int(geometry.intersecting_pixels.size),
        "n_pixels_with_centers_inside_roi": int(
            np.count_nonzero(geometry.center_inside)
        ),
        "pix2vec_vec2pix_roundtrip_mismatches": geometry.n_roundtrip_mismatches,
    }


def ensure_outputs_available(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    map_paths = resolve_map_paths(args)
    output_dir = output_directory(args)
    basename = output_basename(args)
    figure_paths = {
        output_format: output_dir / f"{basename}.{output_format}"
        for output_format in args.formats
    }
    diagnostics_path = output_dir / f"{basename}.json"
    ensure_outputs_available((*figure_paths.values(), diagnostics_path), args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rois = {
        "fragile": args.fragile_roi,
        "resilient": args.resilient_roi,
    }
    geometries = {
        scenario: make_roi_geometry(
            args.nside,
            selected_rois[scenario],
            args.roi_width_deg,
            args.roi_height_deg,
        )
        for scenario in SCENARIOS
    }
    values_by_scenario = {}
    read_diagnostics = {}
    for scenario in SCENARIOS:
        geometry = geometries[scenario]
        print(
            f"Reading {scenario} {geometry.roi_id}: "
            f"l={geometry.frame.longitude_deg:g}, b={geometry.frame.latitude_deg:g}",
            flush=True,
        )
        values_by_scenario[scenario], read_diagnostics[scenario] = (
            read_roi_values(
                map_paths[scenario],
                args.nside,
                geometry.intersecting_pixels,
            )
        )

    rendering = render_comparison(
        figure_paths,
        geometries,
        values_by_scenario,
        args,
    )
    payload = {
        "purpose": "quantitative subhalo and total ROI comparison",
        "equations": {
            "J_sub": "Jpointlike + Jextended",
            "J_tot": "Jsmooth + Jpointlike + Jextended",
        },
        "inputs": {scenario: str(path) for scenario, path in map_paths.items()},
        "repop_id": args.repop_id,
        "native_healpix": {
            "nside": args.nside,
            "ordering": "NESTED",
            "pixel_area_sr": float(hp.nside2pixarea(args.nside)),
            "pixel_area_deg2": float(hp.nside2pixarea(args.nside, degrees=True)),
            "values_are_integrated_j_per_native_pixel": True,
            "resampled": False,
        },
        "roi_dimensions_deg": {
            "width": args.roi_width_deg,
            "height": args.roi_height_deg,
            "convention": (
                "gnomonic tangent-plane bounds: "
                "|x| <= tan(width/2), |y| <= tan(height/2)"
            ),
        },
        "projection_geometry": {
            "x_orientation": "increasing Galactic longitude to the left",
            "longitude_tick_convention": (
                "constant-longitude crossing of the horizontal central axis y=0"
            ),
            "longitude_tick_relation": "x = -cos(b0) tan(delta_l)",
            "latitude_tick_convention": (
                "constant-latitude crossing of the vertical central axis x=0"
            ),
            "latitude_tick_relation": "y = tan(delta_b)",
            "constant_longitude_is_vertical_in_panel": False,
        },
        "scenario_rois": {
            scenario: {
                "geometry": geometry_diagnostics(geometries[scenario], args),
                "read": read_diagnostics[scenario],
            }
            for scenario in SCENARIOS
        },
        "representation": {
            "projection": "gnomonic tangent plane",
            "primitive": "flat-valued projected native HEALPix polygons",
            "smooth_milky_way_component_included_in_J_sub": False,
            "smooth_milky_way_component_included_in_J_tot": True,
            "visual_composition": False,
            "artificial_boost": False,
            "smoothing": False,
            "glow": False,
            "interpolation": False,
            "dilation": False,
            "transparency": False,
        },
        "rendering": rendering,
    }
    diagnostics_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Figure(s): {', '.join(str(path) for path in figure_paths.values())}")
    print(f"Diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
