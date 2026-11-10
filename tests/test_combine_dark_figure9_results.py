import csv
import importlib.util
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "cuts"
    / "scripts"
    / "combine_dark_figure9_results.py"
)
spec = importlib.util.spec_from_file_location(
    "combine_dark_figure9_results", SCRIPT_PATH
)
combine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combine)


def make_row(scenario, repop_id, threshold):
    n_pointlike_total = 1000 + repop_id
    n_extended_total = 200 + repop_id
    kept_fraction = 1.0 - 10.0 * threshold
    n_pointlike_kept = int(n_pointlike_total * kept_fraction)
    n_extended_kept = int(n_extended_total * kept_fraction)
    return {
        "scenario": scenario,
        "repop_id": str(repop_id),
        "input_h5": f"repop_{repop_id:04d}/{scenario}.h5",
        "population_selection": "dark_vmax_le_8kms",
        "max_vmax_km_s": "8",
        "nside": "2048",
        "npix": "50331648",
        "theta_pix_deg": "0.0286328455",
        "theta_min_deg": "0.028633",
        "theta_aperture_deg": "0.034454159",
        "extended_envelope_modes": "theta-s",
        "j_pixel_ref": str(1.0e20 + repop_id * 1.0e17),
        "brightest_pointlike_proxy": "9e19",
        "brightest_extended_proxy": "8e19",
        "pointlike_f": f"{threshold:.17g}",
        "extended_f": f"{threshold:.17g}",
        "pointlike_j_cut": str(threshold * 1.0e20),
        "extended_j_cut": str(threshold * 1.0e20),
        "n_physical_valid_total": str(1300 + 2 * repop_id),
        "n_vmax_excluded": "100",
        "n_valid_total": str(n_pointlike_total + n_extended_total),
        "n_pointlike_total": str(n_pointlike_total),
        "n_pointlike_kept": str(n_pointlike_kept),
        "n_pointlike_discarded": str(
            n_pointlike_total - n_pointlike_kept
        ),
        "n_extended_total": str(n_extended_total),
        "n_extended_kept": str(n_extended_kept),
        "n_extended_discarded": str(
            n_extended_total - n_extended_kept
        ),
        "pointlike_full_js_sum": "3e22",
        "extended_full_js_sum": "2e22",
        "full_js_sum": "5e22",
        "full_combined_max": "1.2e20",
        "discarded_combined_js_sum": str(threshold * 5.0e22),
        "fraction_discarded_js_to_full": str(threshold),
        "max_discarded_combined_theta_s_envelope_pixel": str(
            threshold * 1.0e20
        ),
        "ratio_max_discarded_theta_s_envelope_to_j_pixel_ref": (
            f"{threshold * (1.0 + repop_id / 1000.0):.17g}"
        ),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    paths = {}
    for scenario in combine.SCENARIOS:
        existing = [make_row(scenario, repop_id, 1.0e-3) for repop_id in range(50)]
        additional = [
            make_row(scenario, repop_id, threshold)
            for repop_id in range(50)
            for threshold in combine.ADDITIONAL_THRESHOLDS
        ]
        paths[f"{scenario}_existing"] = tmp_path / f"{scenario}_existing.csv"
        paths[f"{scenario}_additional"] = (
            tmp_path / f"{scenario}_additional.csv"
        )
        write_csv(paths[f"{scenario}_existing"], existing)
        write_csv(paths[f"{scenario}_additional"], additional)
    paths["output"] = tmp_path / "combined.csv"
    return paths


def run_combine(paths):
    combine.combine_results(
        paths["fragile_existing"],
        paths["fragile_additional"],
        paths["resilient_existing"],
        paths["resilient_additional"],
        paths["output"],
    )


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def rewrite(path, transform):
    rows = read_rows(path)
    transform(rows)
    write_csv(path, rows)


def test_successful_aggregation_preserves_plotting_columns_and_metric(inputs):
    run_combine(inputs)
    rows = read_rows(inputs["output"])

    assert len(rows) == 400
    assert combine.PLOTTING_COLUMNS <= set(rows[0])
    assert combine.REFERENCE_RATIO_COLUMN in rows[0]
    assert {
        (row["scenario"], int(row["repop_id"]), float(row["pointlike_f"]))
        for row in rows
    } == {
        (scenario, repop_id, threshold)
        for scenario in combine.SCENARIOS
        for repop_id in range(50)
        for threshold in combine.EXPECTED_THRESHOLDS
    }
    row = next(
        row
        for row in rows
        if row["scenario"] == "fragile"
        and row["repop_id"] == "0"
        and float(row["pointlike_f"]) == 1.0e-3
    )
    assert row[combine.REFERENCE_RATIO_COLUMN] == "0.001"


def test_missing_repopulation_is_rejected(inputs):
    rewrite(
        inputs["fragile_additional"],
        lambda rows: rows.__setitem__(
            slice(None), [row for row in rows if row["repop_id"] != "49"]
        ),
    )
    with pytest.raises(ValueError, match="Wrong repopulations in"):
        run_combine(inputs)


def test_duplicate_key_is_rejected(inputs):
    rewrite(
        inputs["resilient_existing"], lambda rows: rows.append(dict(rows[0]))
    )
    with pytest.raises(ValueError, match="Duplicate .* key"):
        run_combine(inputs)


def test_mismatched_j_pixel_ref_is_rejected(inputs):
    def change_reference(rows):
        rows[0]["j_pixel_ref"] = "2e20"

    rewrite(inputs["fragile_additional"], change_reference)
    with pytest.raises(ValueError, match="'j_pixel_ref' differs"):
        run_combine(inputs)


def test_inconsistent_population_metadata_is_rejected(inputs):
    def change_population(rows):
        rows[0]["population_selection"] = "full"

    rewrite(inputs["resilient_additional"], change_population)
    with pytest.raises(ValueError, match="population_selection"):
        run_combine(inputs)


def test_missing_threshold_combination_is_rejected(inputs):
    def remove_one_threshold(rows):
        rows[:] = [
            row
            for row in rows
            if not (
                row["repop_id"] == "0"
                and float(row["pointlike_f"]) == 1.0e-5
            )
        ]

    rewrite(inputs["resilient_additional"], remove_one_threshold)
    with pytest.raises(ValueError, match="Missing or unexpected"):
        run_combine(inputs)


def test_nonsymmetric_rows_are_not_written(inputs):
    def add_off_diagonal(rows):
        row = dict(rows[0])
        row["extended_f"] = "1e-2"
        rows.append(row)

    rewrite(inputs["fragile_additional"], add_off_diagonal)
    run_combine(inputs)
    assert len(read_rows(inputs["output"])) == 400
