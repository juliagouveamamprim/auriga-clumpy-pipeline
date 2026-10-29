from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "diagnostics" / "maps" / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "explore_auriga_clumpy_rois.py"
sys.path.insert(0, str(SCRIPTS_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preview = load_module("explore_auriga_clumpy_rois", SCRIPT_PATH)


def write_map(path: Path, nside: int, scenario_scale: float) -> dict[str, np.ndarray]:
    npix = hp.nside2npix(nside)
    pixels = np.arange(npix, dtype=np.int64)
    longitude, latitude = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
    phase = 1.0 + (longitude % 360.0) / 360.0 + (latitude + 90.0) / 360.0
    pointlike = scenario_scale * 2.0e14 * phase
    extended = scenario_scale * 7.0e14 * phase**2

    # Exercise the CLUMPY empty-Jextended sentinel and exact-zero rendering.
    extended[::19] = float(np.float32(1.0e-40))
    pointlike[::19] = 0.0
    smooth = 4.0e17 * (1.0 + 0.05 * phase)
    total = smooth + pointlike + extended
    columns = [
        fits.Column(name="PIXEL", format="K", array=pixels),
        # Jtot is deliberately unrelated to the expected displayed value at a
        # few pixels, so a test cannot pass by reading Jtot.
        fits.Column(name="Jtot", format="D", unit="GeV2 cm-5", array=total + 3.0e16),
        fits.Column(name="Jsmooth", format="D", unit="GeV2 cm-5", array=smooth),
        fits.Column(name="Jextended", format="D", unit="GeV2 cm-5", array=extended),
        fits.Column(name="Jpointlike", format="D", unit="GeV2 cm-5", array=pointlike),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="JFACTOR")
    table.header["PIXTYPE"] = "HEALPIX"
    table.header["ORDERING"] = "NESTED"
    table.header["NSIDE"] = nside
    table.header["FIRSTPIX"] = 0
    table.header["LASTPIX"] = npix - 1
    table.header["INDXSCHM"] = "EXPLICIT"
    table.header["COORDSYS"] = "G"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    expected_extended = extended.copy()
    expected_extended[::19] = 0.0
    return {
        "pointlike": pointlike,
        "extended": expected_extended,
        "subhalo": pointlike + expected_extended,
        "smooth": smooth,
        "total": smooth + pointlike + expected_extended,
    }


def test_defaults_are_roi_04_nside2048_viridis_and_20_by_10_degrees():
    args = preview.parse_args([])

    assert args.nside == 2048
    assert args.fragile_roi == "roi_04"
    assert args.resilient_roi == "roi_04"
    assert args.roi_width_deg == pytest.approx(20.0)
    assert args.roi_height_deg == pytest.approx(10.0)
    assert args.cmap == "viridis"


def test_typography_matches_official_all_sky_hierarchy():
    style = preview.plot_style()

    assert style["axes.titlesize"] == 16
    assert style["axes.labelsize"] == 15
    assert style["xtick.labelsize"] == 13
    assert style["ytick.labelsize"] == 13
    assert style["legend.fontsize"] == 12


def test_read_subhalo_values_sums_only_native_scientific_components(tmp_path):
    path = tmp_path / "map.fits"
    expected = write_map(path, nside=8, scenario_scale=1.5)
    pixels = np.array([0, 17, 19, 51], dtype=np.int64)

    values, diagnostics = preview.read_subhalo_values(path, 8, pixels)

    np.testing.assert_array_equal(values, expected["subhalo"][pixels])
    assert diagnostics["source_columns"] == [
        "Jsmooth",
        "Jpointlike",
        "Jextended",
    ]
    assert diagnostics["field_source_columns"]["J_sub"] == [
        "Jpointlike",
        "Jextended",
    ]
    assert diagnostics["equation"] == "J_sub = Jpointlike + Jextended"
    assert diagnostics["read_Jsmooth"] is True
    assert diagnostics["read_Jtot_for_display"] is False
    assert diagnostics["n_empty_extended_sentinel_converted_to_zero"] == 2
    assert diagnostics["n_zero_pixels"] == 2


def test_read_roi_values_constructs_subhalo_and_total_from_components(tmp_path):
    path = tmp_path / "map.fits"
    expected = write_map(path, nside=8, scenario_scale=1.5)
    pixels = np.array([0, 17, 19, 51], dtype=np.int64)

    values, diagnostics = preview.read_roi_values(path, 8, pixels)

    np.testing.assert_array_equal(values["J_sub"], expected["subhalo"][pixels])
    np.testing.assert_array_equal(values["J_tot"], expected["total"][pixels])
    assert diagnostics["source_columns"] == [
        "Jsmooth",
        "Jpointlike",
        "Jextended",
    ]
    assert diagnostics["read_Jsmooth"] is True
    assert diagnostics["read_Jtot_for_display"] is False
    assert diagnostics["field_source_columns"] == {
        "J_sub": ["Jpointlike", "Jextended"],
        "J_tot": ["Jsmooth", "Jpointlike", "Jextended"],
    }
    assert diagnostics["pixel_counts"]["J_sub"]["zero"] == 2
    assert diagnostics["pixel_counts"]["J_tot"]["zero"] == 0


def test_shared_log_norm_uses_exact_joint_positive_range_without_floor():
    values = {
        "fragile": np.array([0.0, 1.0e12, 1.0e14]),
        "resilient": np.array([2.0e13, 1.0e17, 0.0]),
    }

    norm, diagnostics = preview.shared_log_norm(values)

    assert isinstance(norm, LogNorm)
    assert norm.vmin == pytest.approx(1.0e12)
    assert norm.vmax == pytest.approx(1.0e17)
    assert diagnostics["shared_between_scenarios"] is True
    assert diagnostics["percentile_clipping"] is False
    assert diagnostics["artificial_floor"] is False
    assert diagnostics["distributions"]["joint"]["count"] == 4
    assert set(diagnostics["distributions"]["joint"]) == {"count", "min", "max"}


def test_column_log_norms_use_independent_exact_positive_ranges():
    values = {
        "fragile": {
            "J_sub": np.array([0.0, 1.0e12]),
            "J_tot": np.array([1.0e18, 2.0e18]),
        },
        "resilient": {
            "J_sub": np.array([2.0e13, 4.0e14]),
            "J_tot": np.array([3.0e18, 4.0e18]),
        },
    }

    norms, diagnostics = preview.column_log_norms(values)

    assert norms["J_sub"].vmin == pytest.approx(1.0e12)
    assert norms["J_sub"].vmax == pytest.approx(4.0e14)
    assert norms["J_tot"].vmin == pytest.approx(1.0e18)
    assert norms["J_tot"].vmax == pytest.approx(4.0e18)
    assert norms["J_sub"] is not norms["J_tot"]
    assert diagnostics["shared_between_scenarios_within_each_column"] is True
    assert diagnostics["shared_between_columns"] is False
    for field_name, positive_count in (("J_sub", 3), ("J_tot", 4)):
        column = diagnostics["columns"][field_name]
        assert column["percentile_clipping"] is False
        assert column["artificial_floor"] is False
        assert column["distributions"]["joint"]["count"] == positive_count


def test_native_pixel_boundaries_are_projected_as_four_vertex_polygons():
    geometry = preview.make_roi_geometry(16, "roi_04", 20.0, 10.0)

    polygons = preview.projected_pixel_polygons(
        16,
        geometry.intersecting_pixels,
        geometry.frame,
    )

    assert polygons.shape == (geometry.intersecting_pixels.size, 4, 2)
    assert np.all(np.isfinite(polygons))
    assert geometry.n_roundtrip_mismatches == 0


def test_roi_01_direct_and_inverse_gnomonic_transform_roundtrip():
    frame = preview.tangent_frame(10.0, -30.0, 20.0, 10.0)
    projected = np.array(
        [
            [0.0, 0.0],
            [-0.12, -0.04],
            [0.08, 0.06],
        ]
    )
    vectors = np.asarray(
        [
            preview.inverse_projected_vector(x, y, frame)
            for x, y in projected
        ]
    )

    x, y, denominator = preview.project_vectors(vectors, frame)

    np.testing.assert_allclose(np.column_stack((x, y)), projected, atol=1.0e-14)
    assert np.all(denominator > 0.0)


def test_roi_01_longitude_ticks_use_central_axis_meridian_crossings():
    frame = preview.tangent_frame(10.0, -30.0, 20.0, 10.0)
    figure, axes = plt.subplots()

    preview.configure_offset_ticks(axes, frame, 20.0, 10.0)

    longitude_offsets = np.linspace(10.0, -10.0, 5)
    expected_x = -np.cos(np.radians(-30.0)) * np.tan(
        np.radians(longitude_offsets)
    )
    np.testing.assert_allclose(axes.get_xticks(), expected_x)
    assert [label.get_text() for label in axes.get_xticklabels()] == [
        "10",
        "5",
        "0",
        "-5",
        "-10",
    ]
    assert np.all(np.diff(axes.get_xticks()) > 0.0)
    plt.close(figure)


def test_roi_01_central_axis_longitude_relation_matches_direct_projection():
    frame = preview.tangent_frame(10.0, -30.0, 20.0, 10.0)
    longitude_offset = 8.0
    offset_radians = np.radians(longitude_offset)
    central_latitude = np.radians(frame.latitude_deg)
    latitude_on_horizontal_axis = np.degrees(
        np.arctan(np.tan(central_latitude) * np.cos(offset_radians))
    )
    vector = hp.ang2vec(
        frame.longitude_deg + longitude_offset,
        latitude_on_horizontal_axis,
        lonlat=True,
    )

    x, y, denominator = preview.project_vectors(np.atleast_2d(vector), frame)

    assert denominator[0] > 0.0
    assert y[0] == pytest.approx(0.0, abs=1.0e-15)
    assert x[0] == pytest.approx(
        -np.cos(central_latitude) * np.tan(offset_radians)
    )


def test_main_renders_quantitative_subhalo_and_total_roi_04_comparison(
    tmp_path, monkeypatch
):
    fragile_path = tmp_path / "fragile.fits"
    resilient_path = tmp_path / "resilient.fits"
    write_map(fragile_path, nside=16, scenario_scale=0.7)
    write_map(resilient_path, nside=16, scenario_scale=1.4)
    output_dir = tmp_path / "figures"
    panel_titles = []
    panel_title_fontweights = []
    x_labels = []
    y_labels = []
    original_set_title = Axes.set_title
    original_set_xlabel = Axes.set_xlabel
    original_set_ylabel = Axes.set_ylabel

    def record_title(axes, label, *args, **kwargs):
        panel_titles.append(label)
        panel_title_fontweights.append(kwargs.get("fontweight"))
        return original_set_title(axes, label, *args, **kwargs)

    def record_xlabel(axes, label, *args, **kwargs):
        x_labels.append(label)
        return original_set_xlabel(axes, label, *args, **kwargs)

    def record_ylabel(axes, label, *args, **kwargs):
        y_labels.append(label)
        return original_set_ylabel(axes, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", record_title)
    monkeypatch.setattr(Axes, "set_xlabel", record_xlabel)
    monkeypatch.setattr(Axes, "set_ylabel", record_ylabel)

    preview.main(
        [
            "--nside",
            "16",
            "--fragile-map",
            str(fragile_path),
            "--resilient-map",
            str(resilient_path),
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
            "--dpi",
            "100",
        ]
    )

    basename = "subhalo_total_fragile_roi_04_resilient_roi_04_nside16"
    assert (output_dir / f"{basename}.png").is_file()
    payload = json.loads((output_dir / f"{basename}.json").read_text())
    assert payload["equations"] == {
        "J_sub": "Jpointlike + Jextended",
        "J_tot": "Jsmooth + Jpointlike + Jextended",
    }
    assert payload["native_healpix"]["values_are_integrated_j_per_native_pixel"]
    assert payload["native_healpix"]["resampled"] is False
    assert payload["roi_dimensions_deg"] == {
        "width": 20.0,
        "height": 10.0,
        "convention": (
            "gnomonic tangent-plane bounds: "
            "|x| <= tan(width/2), |y| <= tan(height/2)"
        ),
    }
    assert payload["projection_geometry"] == {
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
    }
    assert payload["rendering"]["colormap"] == "viridis"
    assert payload["rendering"]["normalization"]["type"] == (
        "matplotlib.colors.LogNorm"
    )
    assert payload["rendering"]["normalization"][
        "shared_between_scenarios_within_each_column"
    ] is True
    assert payload["rendering"]["normalization"][
        "shared_between_columns"
    ] is False
    assert set(payload["rendering"]["normalization"]["columns"]) == {
        "J_sub",
        "J_tot",
    }
    colorbars = payload["rendering"]["quantitative_colorbars"]
    assert colorbars["count"] == 2
    assert colorbars["orientation"] == "horizontal"
    assert colorbars["one_below_each_column"] is True
    assert colorbars["units"] == "GeV2 cm-5 per native HEALPix pixel"
    assert [item["column"] for item in colorbars["columns"]] == [
        "J_sub",
        "J_tot",
    ]
    for item in colorbars["columns"]:
        normalization = payload["rendering"]["normalization"]["columns"][
            item["column"]
        ]
        assert item["vmin"] == pytest.approx(normalization["vmin"])
        assert item["vmax"] == pytest.approx(normalization["vmax"])
        assert len(item["ticks"]) == 4
        assert item["ticks"][0] == pytest.approx(item["vmin"])
        assert item["ticks"][-1] == pytest.approx(item["vmax"])
    assert payload["rendering"]["all_intersecting_pixels_rendered"] is True
    assert payload["rendering"]["zero_value_rendering"] == {
        "color": "#000000",
        "opaque": True,
        "reason": "zero is outside the domain of LogNorm",
        "positive_floor_inserted": False,
    }
    assert payload["rendering"]["panel_layout"] == {
        "shape": [2, 2],
        "rows": ["fragile", "resilient"],
        "columns": ["J_sub", "J_tot"],
        "same_roi_within_each_scenario_row": True,
    }
    assert panel_titles == [
        "Fragile",
        "Fragile",
        "Resilient",
        "Resilient",
    ]
    assert panel_title_fontweights == ["bold"] * 4
    assert [label for label in x_labels if label] == [
        r"Galactic longitude offset $\Delta\ell$ [deg]",
        r"Galactic longitude offset $\Delta\ell$ [deg]",
        r"$J_{\mathrm{sub}}\;[\mathrm{GeV^2\,cm^{-5}}]$",
        r"$J_{\mathrm{tot}}\;[\mathrm{GeV^2\,cm^{-5}}]$",
    ]
    assert y_labels == [
        r"Galactic latitude offset $\Delta b$ [deg]",
        r"Galactic latitude offset $\Delta b$ [deg]",
    ]
    assert payload["rendering"]["layout_validation"]["suptitle_present"] is False
    assert (
        payload["rendering"]["layout_validation"]["panel_titles_single_line"]
        is True
    )
    assert payload["rendering"]["layout_validation"][
        "colorbars_overlap_panels"
    ] is False
    assert payload["rendering"]["layout_validation"][
        "colorbars_overlap_each_other"
    ] is False
    assert payload["rendering"]["layout_validation"][
        "colorbar_tick_labels_overlap"
    ] is False
    assert payload["rendering"]["layout_validation"][
        "zero_indicator_present"
    ] is True
    assert payload["rendering"]["layout_validation"][
        "zero_indicator_overlaps_colorbars"
    ] is False
    assert payload["rendering"]["layout_validation"][
        "zero_indicator_overlaps_panels"
    ] is False
    assert payload["representation"] == {
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
    }
    for scenario in preview.SCENARIOS:
        scenario_payload = payload["scenario_rois"][scenario]
        assert scenario_payload["geometry"]["roi_id"] == "roi_04"
        assert scenario_payload["read"]["source_columns"] == [
            "Jsmooth",
            "Jpointlike",
            "Jextended",
        ]
        n_pixels = scenario_payload["geometry"]["n_pixels_intersecting_roi"]
        assert payload["rendering"]["rendered_native_pixels"][scenario] == {
            "J_sub": n_pixels,
            "J_tot": n_pixels,
        }


def test_scenarios_can_select_different_rois(tmp_path):
    fragile_path = tmp_path / "fragile.fits"
    resilient_path = tmp_path / "resilient.fits"
    write_map(fragile_path, nside=8, scenario_scale=1.0)
    write_map(resilient_path, nside=8, scenario_scale=2.0)
    output_dir = tmp_path / "different_rois"

    preview.main(
        [
            "--nside",
            "8",
            "--fragile-map",
            str(fragile_path),
            "--resilient-map",
            str(resilient_path),
            "--fragile-roi",
            "roi_01",
            "--resilient-roi",
            "roi_04",
            "--output-dir",
            str(output_dir),
            "--formats",
            "png",
            "--dpi",
            "100",
        ]
    )

    basename = "subhalo_total_fragile_roi_01_resilient_roi_04_nside8"
    payload = json.loads((output_dir / f"{basename}.json").read_text())
    assert payload["scenario_rois"]["fragile"]["geometry"]["roi_id"] == "roi_01"
    assert payload["scenario_rois"]["resilient"]["geometry"]["roi_id"] == "roi_04"
    assert payload["scenario_rois"]["fragile"]["geometry"][
        "center_longitude_deg"
    ] == pytest.approx(10.0)
    assert payload["scenario_rois"]["resilient"]["geometry"][
        "center_longitude_deg"
    ] == pytest.approx(340.0)
