from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "corrections"
    / "scripts"
    / "plot_consolidated_correction_diagnostics.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diagnostic = load_module("plot_consolidated_correction_diagnostics", SCRIPT_PATH)


def make_table(
    scenario,
    *,
    repop_id=(0, 0, 1, 1),
    theta=(1.0, 2.0, 10.0, 20.0),
    r_raw=(0.1, 1.0, 10.0, 100.0),
    r_corr=(0.99, 1.0, 1.01, 1.001),
):
    repop_id = np.asarray(repop_id)
    theta = np.asarray(theta, dtype=np.float64)
    r_raw = np.asarray(r_raw, dtype=np.float64)
    r_corr = np.asarray(r_corr, dtype=np.float64)
    count = len(theta)
    j_s = np.full(count, 100.0)
    rho_s_raw = np.full(count, 4.0)
    rho_ratio = 1.0 / np.sqrt(r_raw)
    return pd.DataFrame(
        {
            "scenario": [scenario] * count,
            "repop_id": repop_id,
            "halo_index": np.arange(count),
            "halo_name": [f"hydro_res_{index + 1:05d}" for index in range(count)],
            "theta_s_deg": theta,
            "J_s": j_s,
            "J_raw": r_raw * j_s,
            "R_raw": r_raw,
            "rho_s_raw": rho_s_raw,
            "rho_s_corr": rho_s_raw * rho_ratio,
            "rho_corr_over_raw": rho_ratio,
            "J_corr": r_corr * j_s,
            "R_corr": r_corr,
            "npix_raw": np.arange(count) + 1,
            "npix_corr": np.arange(count) + 2,
        },
        columns=diagnostic.INPUT_COLUMNS,
    )


def write_table(path, scenario, **kwargs):
    table = make_table(scenario, **kwargs)
    table.to_csv(path, index=False)
    return path


def load_pair(tmp_path):
    fragile_path = write_table(tmp_path / "fragile.csv", "fragile")
    resilient_path = write_table(
        tmp_path / "resilient.csv",
        "resilient",
        repop_id=(0, 1, 2),
        theta=(0.5, 5.0, 50.0),
        r_raw=(0.2, 2.0, 20.0),
        r_corr=(0.999, 1.0, 1.002),
    )
    return {
        "fragile": diagnostic.load_scenario_csv(fragile_path, "fragile"),
        "resilient": diagnostic.load_scenario_csv(resilient_path, "resilient"),
    }


def test_loads_both_scenarios_using_only_plotting_columns(tmp_path):
    data = load_pair(tmp_path)

    assert set(data) == {"fragile", "resilient"}
    assert data["fragile"].scenario == "fragile"
    assert data["resilient"].scenario == "resilient"
    np.testing.assert_array_equal(data["fragile"].repop_id, [0, 0, 1, 1])
    np.testing.assert_allclose(data["resilient"].theta_s_deg, [0.5, 5.0, 50.0])


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_invalid_schema_is_rejected(tmp_path, mutation):
    table = make_table("fragile")
    if mutation == "missing":
        table = table.drop(columns="R_corr")
    elif mutation == "extra":
        table["unexpected"] = 1
    else:
        table = table[[*diagnostic.INPUT_COLUMNS[1:], diagnostic.INPUT_COLUMNS[0]]]
    path = tmp_path / f"{mutation}.csv"
    table.to_csv(path, index=False)

    with pytest.raises(diagnostic.InputValidationError, match="schema mismatch"):
        diagnostic.load_scenario_csv(path, "fragile")


@pytest.mark.parametrize(
    "scenarios",
    [["resilient"] * 4, ["fragile"] * 3 + ["resilient"]],
)
def test_wrong_or_mixed_scenario_is_rejected(tmp_path, scenarios):
    table = make_table("fragile")
    table["scenario"] = scenarios
    path = tmp_path / "wrong_scenario.csv"
    table.to_csv(path, index=False)

    with pytest.raises(diagnostic.InputValidationError, match="expected only scenario"):
        diagnostic.load_scenario_csv(path, "fragile")


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_theta_is_rejected(tmp_path, bad):
    table = make_table("fragile")
    table.loc[1, "theta_s_deg"] = bad
    path = tmp_path / "bad_theta.csv"
    table.to_csv(path, index=False)

    with pytest.raises(diagnostic.InputValidationError, match="theta_s_deg"):
        diagnostic.load_scenario_csv(path, "fragile")


@pytest.mark.parametrize("column", ["R_raw", "R_corr"])
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_ratio_is_rejected(tmp_path, column, bad):
    table = make_table("fragile")
    table.loc[1, column] = bad
    path = tmp_path / f"bad_{column}.csv"
    table.to_csv(path, index=False)

    with pytest.raises(diagnostic.InputValidationError, match=column):
        diagnostic.load_scenario_csv(path, "fragile")


@pytest.mark.parametrize("column", ["theta_s_deg", "R_raw", "R_corr"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_plotting_value_is_rejected(tmp_path, column, bad):
    table = make_table("fragile")
    table.loc[2, column] = bad
    path = tmp_path / f"nonfinite_{column}.csv"
    table.to_csv(path, index=False)

    with pytest.raises(diagnostic.InputValidationError, match=column):
        diagnostic.load_scenario_csv(path, "fragile")


@pytest.mark.parametrize("bad", [-1, 1.5, np.nan, np.inf])
def test_invalid_repopulation_id_is_rejected(tmp_path, bad):
    table = make_table("fragile")
    table.loc[0, "repop_id"] = bad
    path = tmp_path / "bad_repop.csv"
    table.to_csv(path, index=False)

    with pytest.raises(diagnostic.InputValidationError, match="repop_id"):
        diagnostic.load_scenario_csv(path, "fragile")


def test_log10_ratio_calculation(tmp_path):
    path = write_table(
        tmp_path / "fragile.csv",
        "fragile",
        repop_id=(0, 0, 0),
        theta=(1.0, 2.0, 3.0),
        r_raw=(0.01, 1.0, 100.0),
        r_corr=(0.1, 1.0, 10.0),
    )
    data = diagnostic.load_scenario_csv(path, "fragile")

    np.testing.assert_allclose(diagnostic.log10_ratios(data, "raw"), [-2, 0, 2])
    np.testing.assert_allclose(
        diagnostic.log10_ratios(data, "corrected"), [-1, 0, 1]
    )


def test_common_theta_edges_span_both_scenarios(tmp_path):
    data = load_pair(tmp_path)

    edges = diagnostic.common_theta_edges(data, 4)

    assert len(edges) == 5
    assert edges[0] == pytest.approx(0.5)
    assert edges[-1] == pytest.approx(50.0)
    np.testing.assert_allclose(np.diff(np.log10(edges)), 0.5)


def test_known_binned_statistics_counts_repops_and_empty_bins(tmp_path):
    fragile_path = write_table(
        tmp_path / "fragile.csv",
        "fragile",
        repop_id=(0, 1, 1),
        theta=(1.0, 2.0, 20.0),
        r_raw=(1.0, 100.0, 10000.0),
        r_corr=(1.0, 10.0, 100.0),
    )
    resilient_path = write_table(
        tmp_path / "resilient.csv",
        "resilient",
        repop_id=(0, 0),
        theta=(5.0, 50.0),
        r_raw=(10.0, 1000.0),
        r_corr=(1.0, 1.0),
    )
    data = {
        "fragile": diagnostic.load_scenario_csv(fragile_path, "fragile"),
        "resilient": diagnostic.load_scenario_csv(resilient_path, "resilient"),
    }
    edges = np.asarray([1.0, 10.0, 100.0, 1000.0])

    binned = diagnostic.build_binned_theta_summary(data, edges)

    first = binned.loc[
        (binned["scenario"] == "fragile")
        & (binned["stage"] == "raw")
        & (binned["theta_lo_deg"] == 1.0)
    ].iloc[0]
    assert first["n_halos"] == 2
    assert first["n_repopulations"] == 2
    assert first["median_log10_R"] == pytest.approx(1.0)
    assert first["p16_log10_R"] == pytest.approx(0.32)
    assert first["p84_log10_R"] == pytest.approx(1.68)

    empty = binned.loc[
        (binned["scenario"] == "fragile")
        & (binned["stage"] == "corrected")
        & (binned["theta_lo_deg"] == 100.0)
    ].iloc[0]
    assert empty["n_halos"] == 0
    assert empty["n_repopulations"] == 0
    assert np.isnan(empty["median_log10_R"])
    assert np.isnan(empty["p16_log10_R"])
    assert np.isnan(empty["p84_log10_R"])


def test_histograms_are_normalized_independently_for_unequal_samples():
    edges = np.asarray([-1.0, 0.0, 1.0])
    fragile = np.asarray([-0.5, 0.5])
    resilient = np.full(1000, 0.5)

    fragile_fractions = diagnostic.normalized_histogram(fragile, edges)
    resilient_fractions = diagnostic.normalized_histogram(resilient, edges)

    np.testing.assert_allclose(fragile_fractions, [0.5, 0.5])
    np.testing.assert_allclose(resilient_fractions, [0.0, 1.0])
    assert np.sum(fragile_fractions) == pytest.approx(1.0)
    assert np.sum(resilient_fractions) == pytest.approx(1.0)


def test_distribution_figure_labels_fraction_and_stage_specific_ratios(
    tmp_path, monkeypatch
):
    data = load_pair(tmp_path)
    captured = {}

    def capture_figure(figure, output_dir, stem, formats):
        captured["figure"] = figure
        return []

    monkeypatch.setattr(diagnostic, "_save_figure", capture_figure)
    diagnostic.make_distribution_figure(data, tmp_path, ("png",))
    figure = captured["figure"]
    try:
        assert [axis.get_ylabel() for axis in figure.axes] == [
            "Fraction of halos",
            "Fraction of halos",
        ]
        assert [axis.get_xlabel() for axis in figure.axes] == [
            r"$\log_{10}(J_{\rm raw}/J_s)$",
            r"$\log_{10}(J_{\rm corr}/J_s)$",
        ]
    finally:
        diagnostic.plt.close(figure)


def test_normalized_histogram_does_not_request_density(monkeypatch):
    original_histogram = diagnostic.np.histogram
    calls = []

    def tracked_histogram(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_histogram(*args, **kwargs)

    monkeypatch.setattr(diagnostic.np, "histogram", tracked_histogram)
    fractions = diagnostic.normalized_histogram(
        np.asarray([-0.5, 0.5]), np.asarray([-1.0, 0.0, 1.0])
    )

    np.testing.assert_allclose(fractions, [0.5, 0.5])
    assert len(calls) == 1
    assert calls[0].get("density", False) is False


def test_histogram_range_includes_equality_reference():
    positive = diagnostic.common_histogram_edges([np.asarray([0.2, 0.4])], 4)
    negative = diagnostic.common_histogram_edges([np.asarray([-0.4, -0.2])], 4)

    assert positive[0] == 0.0
    assert negative[-1] == 0.0


def test_main_writes_exact_figures_and_derived_csvs_from_synthetic_inputs(tmp_path):
    fragile_path = write_table(tmp_path / "fragile.csv", "fragile")
    resilient_path = write_table(
        tmp_path / "resilient.csv",
        "resilient",
        repop_id=(0, 1, 2),
        theta=(0.5, 5.0, 50.0),
        r_raw=(0.2, 2.0, 20.0),
        r_corr=(0.999, 1.0, 1.002),
    )
    output_dir = tmp_path / "plots"

    status = diagnostic.main(
        [
            "--fragile-csv",
            str(fragile_path),
            "--resilient-csv",
            str(resilient_path),
            "--output-dir",
            str(output_dir),
            "--theta-bins",
            "4",
        ]
    )

    assert status == 0
    for filename in (
        "correction_log_ratio_distributions.png",
        "correction_log_ratio_distributions.pdf",
        "correction_log_ratio_vs_theta.png",
        "correction_log_ratio_vs_theta.pdf",
    ):
        path = output_dir / filename
        assert path.is_file()
        assert path.stat().st_size > 0

    summary_path = output_dir / "data" / "correction_distribution_summary.csv"
    binned_path = output_dir / "data" / "correction_ratio_vs_theta_binned.csv"
    summary = pd.read_csv(summary_path)
    binned = pd.read_csv(binned_path)
    assert list(summary.columns) == diagnostic.SUMMARY_COLUMNS
    assert list(binned.columns) == diagnostic.BINNED_COLUMNS
    assert len(summary) == 4
    assert len(binned) == 2 * 2 * 4
    assert set(summary["scenario"]) == {"fragile", "resilient"}
    assert set(summary["stage"]) == {"raw", "corrected"}


def test_defaults_do_not_require_real_csvs_until_execution():
    args = diagnostic.parse_args([])

    assert args.fragile_csv.name == "corrections_fragile_repop0000_0021.csv"
    assert args.resilient_csv.name == "corrections_resilient_repop0000_0021.csv"
    assert args.theta_bins == 24
    assert args.formats == ("png", "pdf")
