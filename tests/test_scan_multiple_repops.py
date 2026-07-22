import csv
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
    / "scan_multiple_repops.py"
)

spec = importlib.util.spec_from_file_location(
    "scan_multiple_repops",
    SCRIPT_PATH,
)
multi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi)


def make_row(repop_id, central, aperture, theta_s, n_pl, n_ext):
    return {
        "scenario": "resilient",
        "repop_id": str(repop_id),
        "nside": "2048",
        "pointlike_f": "1e-3",
        "extended_f": "1e-3",
        "theta_aperture_deg": "0.015",
        "extended_envelope_modes": "aperture,theta-s",
        "n_valid_total": "1000",
        "n_pointlike_kept": str(n_pl),
        "n_extended_kept": str(n_ext),
        "ratio_max_discarded_to_final": str(central),
        "ratio_max_discarded_aperture_envelope_to_final": str(
            aperture
        ),
        "ratio_max_discarded_theta_s_envelope_to_final": str(
            theta_s
        ),
    }


def test_parsers_and_individual_stem():
    assert multi.parse_scenarios(
        "fragile,resilient,fragile"
    ) == ["fragile", "resilient"]

    assert multi.parse_envelope_modes(
        "aperture,theta-s,aperture"
    ) == ["aperture", "theta-s"]

    assert multi.individual_stem(
        7,
        "fragile",
        2048,
        ["aperture", "theta-s"],
    ) == "repop_0007_fragile_nside2048_envelopes_scan"

    with pytest.raises(ValueError):
        multi.parse_scenarios("invalid")


def test_aggregate_rows_computes_dispersion_and_worst_repop():
    rows = [
        make_row(
            repop_id=0,
            central=0.002,
            aperture=0.003,
            theta_s=0.006,
            n_pl=100,
            n_ext=50,
        ),
        make_row(
            repop_id=1,
            central=0.004,
            aperture=0.005,
            theta_s=0.008,
            n_pl=120,
            n_ext=40,
        ),
    ]

    enriched = multi.add_derived_columns(
        rows,
        Path("individual.csv"),
    )
    summary = multi.aggregate_rows(enriched)

    assert len(summary) == 1

    result = summary[0]

    assert result["n_repops"] == 2
    assert np.isclose(
        result["ratio_max_discarded_to_final_mean"],
        0.003,
    )
    assert np.isclose(
        result["ratio_max_discarded_to_final_std"],
        np.std([0.002, 0.004], ddof=1),
    )
    assert result[
        "ratio_max_discarded_to_final_max_repop_id"
    ] == 1

    theta_key = (
        "ratio_max_discarded_theta_s_envelope_to_final"
    )
    assert np.isclose(result[f"{theta_key}_max"], 0.008)
    assert result[f"{theta_key}_max_repop_id"] == 1

    assert np.isclose(result["n_total_kept_mean"], 155.0)


def test_validate_existing_csv(tmp_path):
    path = tmp_path / "individual.csv"

    rows = [
        {
            "repop_id": 0,
            "scenario": "fragile",
            "nside": 2048,
            "extended_envelope_modes": "aperture,theta-s",
            "pointlike_f": "1e-3",
            "extended_f": "1e-3",
            "theta_aperture_deg": "0.015",
        }
    ]

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    validated = multi.validate_existing_csv(
        path=path,
        repop_id=0,
        scenario="fragile",
        nside=2048,
        pointlike_f_values=[1e-3],
        extended_f_values=[1e-3],
        envelope_modes=["aperture", "theta-s"],
        theta_aperture_deg=0.015,
    )

    assert len(validated) == 1

    with pytest.raises(ValueError):
        multi.validate_existing_csv(
            path=path,
            repop_id=0,
            scenario="resilient",
            nside=2048,
            pointlike_f_values=[1e-3],
            extended_f_values=[1e-3],
            envelope_modes=["aperture", "theta-s"],
            theta_aperture_deg=0.015,
        )
