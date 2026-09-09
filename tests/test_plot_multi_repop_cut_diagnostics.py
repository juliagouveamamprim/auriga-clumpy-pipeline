import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "cuts"
    / "scripts"
    / "plot_multi_repop_cut_diagnostics.py"
)

spec = importlib.util.spec_from_file_location(
    "plot_multi_repop_cut_diagnostics",
    SCRIPT_PATH,
)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_prepare_diagonal_uses_j_pixel_reference_ratio():
    table = pd.DataFrame(
        [
            {
                "scenario": "resilient",
                "repop_id": 0,
                "nside": 2048,
                "pointlike_f": 1.0e-3,
                "extended_f": 1.0e-3,
                "j_pixel_ref": 10.0,
                (
                    "ratio_max_discarded_theta_s_envelope_"
                    "to_j_pixel_ref"
                ): 0.006,
                "fraction_discarded_js_to_full": 0.2,
                "n_pointlike_total": 100,
                "n_pointlike_kept": 80,
                "n_pointlike_discarded": 20,
                "n_extended_total": 50,
                "n_extended_kept": 30,
                "n_extended_discarded": 20,
            }
        ]
    )

    plot.validate_columns(table)
    prepared = plot.add_reference_ratio_fallback(table)
    diagonal = plot.prepare_diagonal(prepared)

    assert np.isclose(diagonal.iloc[0]["impact_percent"], 0.6)


def test_old_combined_csv_derives_reference_ratio():
    table = pd.DataFrame(
        {
            "j_pixel_ref": [10.0],
            "max_discarded_combined_theta_s_envelope_pixel": [0.06],
        }
    )

    prepared = plot.add_reference_ratio_fallback(table)

    assert np.isclose(
        prepared.iloc[0][plot.REFERENCE_RATIO_COLUMN],
        0.006,
    )


@pytest.mark.parametrize("j_pixel_ref", [0.0, np.nan, np.inf])
def test_reference_ratio_rejects_invalid_denominator(j_pixel_ref):
    table = pd.DataFrame(
        {
            "j_pixel_ref": [j_pixel_ref],
            "max_discarded_combined_theta_s_envelope_pixel": [0.06],
        }
    )

    with pytest.raises(ValueError, match="finite and positive"):
        plot.add_reference_ratio_fallback(table)
