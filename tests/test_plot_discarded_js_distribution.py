import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "cuts"
    / "scripts"
    / "plot_discarded_js_distribution.py"
)

spec = importlib.util.spec_from_file_location(
    "plot_discarded_js_distribution",
    SCRIPT_PATH,
)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_cli_defaults_to_point_one_dex_rebinning(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--input-npz",
            "input.npz",
            "--output-dir",
            "plots",
        ],
    )

    assert plot.parse_args().rebin_factor == 2
    assert plot.parse_args().population == "combined"


@pytest.mark.parametrize("population", ["all", "pointlike", "extended"])
def test_cli_preserves_individual_population_modes(monkeypatch, population):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--input-npz",
            "input.npz",
            "--output-dir",
            "plots",
            "--population",
            population,
        ],
    )

    assert plot.parse_args().population == population


def test_rebin_histograms_sums_adjacent_bins():
    histograms = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=np.int64,
    )
    edges = np.linspace(-0.4, 0.0, 9)

    rebinned, rebinned_edges = plot.rebin_histograms(
        histograms,
        edges,
        rebin_factor=2,
    )

    np.testing.assert_array_equal(
        rebinned,
        [[1, 5, 9, 13], [17, 21, 25, 29]],
    )
    np.testing.assert_allclose(
        rebinned_edges,
        [-0.4, -0.3, -0.2, -0.1, 0.0],
    )


def test_aggregate_repops_rebins_before_percentage_and_sums_to_100():
    histograms = np.array(
        [
            [1, 1, 1, 1, 3, 3, 3, 3],
            [2, 2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.int64,
    )
    totals = np.array([16, 16], dtype=np.int64)
    edges = np.linspace(-0.4, 0.0, 9)

    result = plot.aggregate_repop_distributions(
        histograms=histograms,
        totals=totals,
        log_relative_edges=edges,
        rebin_factor=2,
    )

    expected_percentage = np.array(
        [
            [12.5, 12.5, 37.5, 37.5],
            [25.0, 25.0, 25.0, 25.0],
        ]
    )
    np.testing.assert_allclose(
        result["log_relative_edges"],
        [-0.4, -0.3, -0.2, -0.1, 0.0],
    )
    np.testing.assert_allclose(result["percentage"], expected_percentage)
    np.testing.assert_allclose(
        np.sum(result["percentage"], axis=1),
        [100.0, 100.0],
    )
    np.testing.assert_allclose(result["mean"], [18.75, 18.75, 31.25, 31.25])
    np.testing.assert_allclose(
        result["lower"],
        np.percentile(expected_percentage, 16.0, axis=0),
    )
    np.testing.assert_allclose(
        result["upper"],
        np.percentile(expected_percentage, 84.0, axis=0),
    )


def test_aggregate_percentiles_include_zero_density_catalogues():
    histograms = np.zeros((10, 4), dtype=np.int64)
    histograms[-1, :2] = 1
    totals = np.ones(10, dtype=np.int64)
    totals[-1] = 2

    result = plot.aggregate_repop_distributions(
        histograms=histograms,
        totals=totals,
        log_relative_edges=np.linspace(-0.2, 0.0, 5),
        rebin_factor=2,
    )

    np.testing.assert_allclose(result["percentage"][:-1], 0.0)
    np.testing.assert_allclose(result["percentage"][-1], [100.0, 0.0])
    np.testing.assert_allclose(result["mean"], [10.0, 0.0])
    np.testing.assert_allclose(result["lower"], [0.0, 0.0])
    np.testing.assert_allclose(result["upper"], [0.0, 0.0])


def test_percentage_axis_label_omits_rebinned_width():
    assert plot.percentage_axis_label(np.linspace(-0.4, 0.0, 5)) == (
        r"Fraction of discarded subhalos [\%]"
    )
    assert plot.percentage_axis_label(np.linspace(-0.4, 0.0, 3)) == (
        r"Fraction of discarded subhalos [\%]"
    )


def test_rebin_histograms_requires_divisible_bin_count():
    with pytest.raises(ValueError, match="must be divisible"):
        plot.rebin_histograms(
            np.ones((2, 5)),
            np.linspace(-0.25, 0.0, 6),
            rebin_factor=2,
        )


def test_plot_loader_rejects_incomplete_checkpoint(tmp_path):
    path = tmp_path / "partial.npz"
    payload = {
        "format_version": np.asarray(plot.FORMAT_VERSION),
        "log_relative_edges": np.array([-2.0, -1.0, 0.0]),
        "requested_repop_ids": np.array([0, 1]),
        "requested_scenarios": np.array(["fragile"]),
        "catalogue_repop_ids": np.array([0]),
        "catalogue_scenarios": np.array(["fragile"]),
    }
    for population in plot.POPULATIONS:
        payload[f"hist_{population}"] = np.array([[1, 1]])
        payload[f"n_discarded_{population}"] = np.array([2])
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="incomplete"):
        plot.load_distribution(path)


def test_plot_draws_step_means_and_edge_aligned_bands(monkeypatch, tmp_path):
    edges = np.linspace(-0.4, 0.0, 9)
    data = {
        "log_relative_edges": edges,
        "catalogue_scenarios": np.array(
            ["fragile", "fragile", "resilient", "resilient"]
        ),
        "hist_all": np.array(
            [
                [1, 1, 1, 1, 3, 3, 3, 3],
                [2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2],
            ]
        ),
        "n_discarded_all": np.array([16, 16, 16, 16]),
    }
    figure = Mock()
    axis = Mock(spec=[
        "stairs",
        "set_xscale",
        "set_xlim",
        "set_xlabel",
        "set_ylabel",
        "tick_params",
        "xaxis",
        "yaxis",
        "grid",
        "legend",
    ])
    axis.xaxis = Mock()
    axis.yaxis = Mock()
    fragile_band_handle = Mock()
    fragile_mean_handle = Mock()
    resilient_band_handle = Mock()
    resilient_mean_handle = Mock()
    axis.stairs.side_effect = [
        fragile_band_handle,
        fragile_mean_handle,
        resilient_band_handle,
        resilient_mean_handle,
    ]
    monkeypatch.setattr(plot.plt, "subplots", Mock(return_value=(figure, axis)))
    monkeypatch.setattr(plot.plt, "close", Mock())

    saved = plot.plot_distribution(
        data=data,
        population="all",
        output_dir=tmp_path,
        formats=("png", "pdf"),
        dpi=180,
        rebin_factor=2,
    )

    rebinned_edges = 10.0 ** np.array([-0.4, -0.3, -0.2, -0.1, 0.0])
    assert axis.stairs.call_count == 4
    (
        fragile_band_call,
        fragile_mean_call,
        resilient_band_call,
        resilient_mean_call,
    ) = axis.stairs.call_args_list
    fragile_percentage = np.array(
        [
            [12.5, 12.5, 37.5, 37.5],
            [25.0, 25.0, 25.0, 25.0],
        ]
    )
    np.testing.assert_allclose(
        fragile_band_call.args[0],
        np.percentile(fragile_percentage, 84.0, axis=0),
    )
    np.testing.assert_allclose(fragile_band_call.args[1], rebinned_edges)
    np.testing.assert_allclose(
        fragile_band_call.kwargs["baseline"],
        np.percentile(fragile_percentage, 16.0, axis=0),
    )
    assert {
        key: value
        for key, value in fragile_band_call.kwargs.items()
        if key != "baseline"
    } == {
        "color": "#EE3377",
        "alpha": 0.18,
        "linewidth": 0.0,
        "label": "Fragile bin-wise 16--84 percentile",
        "fill": True,
        "zorder": 1,
    }
    np.testing.assert_allclose(
        fragile_mean_call.args[0],
        [18.75, 18.75, 31.25, 31.25],
    )
    np.testing.assert_allclose(fragile_mean_call.args[1], rebinned_edges)
    assert fragile_mean_call.kwargs == {
        "color": "#EE3377",
        "linewidth": 2.2,
        "label": "Fragile mean",
        "fill": False,
        "zorder": 2,
    }
    resilient_percentage = np.array(
        [
            [37.5, 37.5, 12.5, 12.5],
            [25.0, 25.0, 25.0, 25.0],
        ]
    )
    np.testing.assert_allclose(
        resilient_band_call.args[0],
        np.percentile(resilient_percentage, 84.0, axis=0),
    )
    np.testing.assert_allclose(resilient_band_call.args[1], rebinned_edges)
    np.testing.assert_allclose(
        resilient_band_call.kwargs["baseline"],
        np.percentile(resilient_percentage, 16.0, axis=0),
    )
    assert {
        key: value
        for key, value in resilient_band_call.kwargs.items()
        if key != "baseline"
    } == {
        "color": "#009988",
        "alpha": 0.18,
        "linewidth": 0.0,
        "label": "Resilient bin-wise 16--84 percentile",
        "fill": True,
        "zorder": 1,
    }
    np.testing.assert_allclose(
        resilient_mean_call.args[0],
        [31.25, 31.25, 18.75, 18.75],
    )
    np.testing.assert_allclose(resilient_mean_call.args[1], rebinned_edges)
    assert resilient_mean_call.kwargs == {
        "color": "#009988",
        "linewidth": 2.2,
        "label": "Resilient mean",
        "fill": False,
        "zorder": 2,
    }
    axis.set_xlim.assert_called_once_with(1.0e-12, 1.0e-2)
    axis.set_xlabel.assert_called_once_with(r"$J_s/J_{s,\max}^{\rm cat}$")
    assert plot.plt.rcParams["axes.labelsize"] == 17.0
    axis.set_ylabel.assert_called_once_with(
        r"Fraction of discarded subhalos [\%]",
        fontsize=15,
    )
    axis.tick_params.assert_called_once_with(
        axis="both",
        which="both",
        labelsize=15,
    )
    legend_kwargs = axis.legend.call_args.kwargs
    assert legend_kwargs["handles"] == [
        fragile_mean_handle,
        fragile_band_handle,
        resilient_mean_handle,
        resilient_band_handle,
    ]
    assert legend_kwargs["labels"] == [
        "Fragile mean",
        "Fragile bin-wise 16--84 percentile",
        "Resilient mean",
        "Resilient bin-wise 16--84 percentile",
    ]
    assert legend_kwargs["ncols"] == 1
    assert legend_kwargs["loc"] == "upper right"
    assert legend_kwargs["fontsize"] == 12
    assert legend_kwargs["labelspacing"] == 0.3
    assert legend_kwargs["handlelength"] == 2.0
    assert legend_kwargs["handletextpad"] == 0.5
    assert legend_kwargs["borderaxespad"] == 0.4
    assert "bbox_to_anchor" not in legend_kwargs
    assert legend_kwargs["frameon"] is False
    assert saved == [
        tmp_path / "discarded_js_distribution_all.png",
        tmp_path / "discarded_js_distribution_all.pdf",
    ]


def test_combined_plot_has_shared_two_panel_layout_and_single_legend(
    monkeypatch,
    tmp_path,
):
    edges = np.linspace(-0.4, 0.0, 9)
    data = {
        "log_relative_edges": edges,
        "catalogue_scenarios": np.array(
            ["fragile", "fragile", "resilient", "resilient"]
        ),
    }
    for population, multiplier in (("pointlike", 1), ("extended", 2)):
        data[f"hist_{population}"] = np.array(
            [
                [1, 1, 1, 1, 3, 3, 3, 3],
                [2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2],
            ]
        ) * multiplier
        data[f"n_discarded_{population}"] = np.array([16, 16, 16, 16]) * multiplier

    figure = Mock()
    axes = [Mock(), Mock()]
    for axis in axes:
        axis.xaxis = Mock()
        axis.yaxis = Mock()
        axis.transAxes = Mock()
        axis.stairs.side_effect = [Mock(), Mock(), Mock(), Mock()]
    monkeypatch.setattr(
        plot.plt,
        "subplots",
        Mock(return_value=(figure, np.asarray(axes, dtype=object))),
    )
    monkeypatch.setattr(plot.plt, "close", Mock())

    saved = plot.plot_distribution(
        data=data,
        population="combined",
        output_dir=tmp_path,
        formats=("png", "pdf"),
        dpi=400,
        rebin_factor=2,
    )

    subplots_kwargs = plot.plt.subplots.call_args.kwargs
    assert subplots_kwargs["sharex"] is True
    assert subplots_kwargs["sharey"] is True
    assert subplots_kwargs["figsize"] == (7.5, 8.4)
    assert axes[0].text.call_args.args == (0.02, 0.93, "Pointlike subhalos")
    assert axes[0].text.call_args.kwargs == {
        "transform": axes[0].transAxes,
        "ha": "left",
        "va": "top",
        "fontsize": 16,
    }
    assert axes[1].text.call_args.args == (0.02, 0.93, "Extended subhalos")
    assert axes[1].text.call_args.kwargs == {
        "transform": axes[1].transAxes,
        "ha": "left",
        "va": "top",
        "fontsize": 16,
    }
    assert axes[0].legend.call_count == 1
    assert axes[1].legend.call_count == 0
    assert axes[0].legend.call_args.kwargs["loc"] == "upper right"
    assert axes[0].legend.call_args.kwargs["fontsize"] == 12
    assert axes[0].legend.call_args.kwargs["ncols"] == 1
    axes[0].set_ylim.assert_called_once_with(0.0, 37.275)
    figure.supylabel.assert_called_once_with(
        r"Fraction of discarded subhalos [\%]",
        fontsize=15,
    )
    assert saved == [
        tmp_path / "discarded_js_distribution_pointlike_extended.png",
        tmp_path / "discarded_js_distribution_pointlike_extended.pdf",
    ]
