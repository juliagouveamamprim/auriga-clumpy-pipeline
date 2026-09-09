import importlib.util
from pathlib import Path

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


def test_aggregate_repops_normalizes_before_binwise_statistics():
    histograms = np.array(
        [
            [1, 3],
            [2, 2],
        ],
        dtype=np.int64,
    )
    totals = np.array([4, 4], dtype=np.int64)
    edges = np.array([-2.0, -1.5, -1.0])

    result = plot.aggregate_repop_distributions(
        histograms=histograms,
        totals=totals,
        log_relative_edges=edges,
    )

    expected_density = np.array(
        [
            [0.5, 1.5],
            [1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(result["density"], expected_density)
    np.testing.assert_allclose(result["mean"], [0.75, 1.25])
    np.testing.assert_allclose(
        result["lower"],
        np.percentile(expected_density, 16.0, axis=0),
    )
    np.testing.assert_allclose(
        result["upper"],
        np.percentile(expected_density, 84.0, axis=0),
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
