from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "maps"
    / "scripts"
    / "plot_auriga_clumpy_conservative_decomposition.py"
)


def load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prototype = load_module(
    "plot_auriga_clumpy_conservative_decomposition",
    PROTOTYPE_PATH,
)


def test_default_colormap_is_unchanged():
    source = matplotlib.colormaps["viridis"]
    displayed = prototype.display_colormap("viridis")
    samples = np.linspace(0.0, 1.0, 257)

    assert prototype.parse_args([]).cmap_min_fraction == 0.0
    np.testing.assert_array_equal(displayed(samples), source(samples))


@pytest.mark.parametrize("minimum", [0.20, 0.30])
def test_truncated_colormap_samples_requested_interval(minimum):
    source = matplotlib.colormaps["viridis"]
    displayed = prototype.display_colormap("viridis", minimum)

    np.testing.assert_allclose(displayed(0.0), source(minimum))
    np.testing.assert_allclose(displayed(1.0), source(1.0))
    np.testing.assert_allclose(
        displayed(np.arange(source.N)),
        source(np.linspace(minimum, 1.0, source.N)),
    )


def test_lowest_valid_positive_uses_selected_lower_colormap_fraction():
    minimum = 0.20
    vmin, vmax = 12.0, 18.0
    normalization = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    displayed = prototype.display_colormap("viridis", minimum)

    np.testing.assert_allclose(
        displayed(normalization(vmin)),
        matplotlib.colormaps["viridis"](minimum),
    )


@pytest.mark.parametrize("minimum", [-0.01, 1.0, 1.01, float("nan")])
def test_invalid_colormap_min_fraction_is_rejected(minimum):
    args = prototype.parse_args(["--cmap-min-fraction", str(minimum)])

    with pytest.raises(
        ValueError,
        match=r"--cmap-min-fraction must satisfy 0 <= value < 1",
    ):
        prototype.validate_args(args)


def test_log_rasters_preserve_zero_sentinel_and_outside_mask():
    geometry = prototype.make_conservative_geometry(xsize=8)
    linear_sums = {
        scenario: {
            component.key: np.zeros(geometry.xsize * geometry.ysize)
            for component in prototype.COMPONENTS
        }
        for scenario in prototype.SCENARIOS
    }
    positive_cell = int(geometry.valid_flat_cells[0])
    linear_sums["fragile"]["pointlike"][positive_cell] = 1.0e12

    raster = prototype.make_log_rasters(linear_sums, geometry)["fragile"][
        "pointlike"
    ]
    inside = geometry.inside

    assert raster.ravel()[positive_cell] == 12.0
    assert np.all(raster[inside & (raster != 12.0)] == prototype.ZERO_SENTINEL)
    assert np.all(np.isnan(raster[~inside]))


def test_map_and_colorbars_share_the_same_truncated_colormap(tmp_path, monkeypatch):
    args = prototype.parse_args(
        [
            "--xsize",
            "8",
            "--panel-scale",
            "1",
            "--dpi",
            "72",
            "--cmap-min-fraction",
            "0.2",
            "--no-graticule",
        ]
    )
    geometry = prototype.make_conservative_geometry(args.xsize)
    raster = np.full(geometry.inside.shape, np.nan)
    raster[geometry.inside] = prototype.ZERO_SENTINEL
    raster.ravel()[geometry.valid_flat_cells[0]] = 1.0
    raster.ravel()[geometry.valid_flat_cells[-1]] = 2.0
    projected = {
        scenario: {component.key: raster for component in prototype.COMPONENTS}
        for scenario in prototype.SCENARIOS
    }
    limits = {component.key: (1.0, 2.0) for component in prototype.COMPONENTS}
    map_colormaps = []
    colorbar_colormaps = []
    rendered_layers = []
    original_imshow = matplotlib.axes.Axes.imshow
    original_colorbar = matplotlib.figure.Figure.colorbar

    def record_imshow(self, *imshow_args, **imshow_kwargs):
        rendered_layers.append((imshow_args[0], imshow_kwargs["cmap"]))
        if isinstance(imshow_kwargs.get("norm"), matplotlib.colors.Normalize):
            map_colormaps.append(imshow_kwargs["cmap"])
        return original_imshow(self, *imshow_args, **imshow_kwargs)

    def record_colorbar(self, mappable, *colorbar_args, **colorbar_kwargs):
        colorbar_colormaps.append(mappable.cmap)
        return original_colorbar(self, mappable, *colorbar_args, **colorbar_kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", record_imshow)
    monkeypatch.setattr(matplotlib.figure.Figure, "colorbar", record_colorbar)

    prototype.render_figure(
        "fragile",
        projected,
        limits,
        tmp_path / "preview.png",
        args,
    )

    assert (
        len(map_colormaps)
        == len(colorbar_colormaps)
        == len(prototype.COMPONENTS)
    )
    assert all(cmap is map_colormaps[0] for cmap in map_colormaps)
    assert all(cmap is map_colormaps[0] for cmap in colorbar_colormaps)
    zero_layer, zero_cmap = rendered_layers[0]
    positive_layer, _ = rendered_layers[1]
    zero_mask = np.ma.getmaskarray(zero_layer)
    positive_mask = np.ma.getmaskarray(positive_layer)
    sentinel = raster == prototype.ZERO_SENTINEL
    outside = ~geometry.inside

    assert np.all(~zero_mask[geometry.inside])
    assert np.all(zero_mask[outside])
    assert np.all(positive_mask[sentinel])
    assert np.all(positive_mask[outside])
    np.testing.assert_allclose(
        zero_cmap(0.0),
        matplotlib.colors.to_rgba(args.zero_color),
    )
