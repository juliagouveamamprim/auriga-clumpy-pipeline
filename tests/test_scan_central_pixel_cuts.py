import argparse
import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "cuts"
    / "scripts"
    / "scan_central_pixel_cuts.py"
)

spec = importlib.util.spec_from_file_location(
    "scan_central_pixel_cuts",
    SCRIPT_PATH,
)
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def test_extended_envelope_repeats_proxy_over_footprint(monkeypatch):
    target_maps = {
        "theta-s": {
            1e-3: np.zeros(4, dtype=float),
            1e-2: np.zeros(4, dtype=float),
        }
    }
    cuts = {1e-3: 0.1, 1e-2: 1.0}

    monkeypatch.setattr(
        scan.hp,
        "ang2vec",
        lambda *_args, **_kwargs: np.array([[1.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(
        scan.hp,
        "query_disc",
        lambda *_args, **_kwargs: np.array([1, 3]),
    )

    scan.add_extended_envelope_maps(
        target_maps=target_maps,
        cuts=cuts,
        nside=1,
        lon_deg=np.array([0.0]),
        lat_deg=np.array([0.0]),
        theta_s_deg=np.array([1.0]),
        weights=np.array([0.5]),
    )

    maps = target_maps["theta-s"]
    np.testing.assert_allclose(maps[1e-3], 0.0)
    np.testing.assert_allclose(
        maps[1e-2],
        [0.0, 0.5, 0.0, 0.5],
    )


def test_discarded_peak_is_normalized_by_catalogue_reference(
    monkeypatch,
    tmp_path,
):
    input_h5 = tmp_path / "synthetic.h5"
    output_csv = tmp_path / "scan.csv"
    column_names = [
        "Js",
        "D_Earth",
        "theta_s",
        "r_s",
        "rho_s",
        "Xearth",
        "Yearth",
        "Zearth",
    ]
    data = np.array(
        [
            [100.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [100.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [30.0, 10.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0],
            [30.0, 10.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    with h5py.File(input_h5, "w") as h5:
        group = h5.create_group("iteration_0")
        dataset = group.create_dataset("data", data=data)
        dataset.attrs["column_names"] = column_names

    args = argparse.Namespace(
        repop_id=0,
        scenario="resilient",
        input_h5=input_h5,
        nside=1,
        pointlike_f_values="0.5",
        extended_f_values="0.5",
        theta_aperture_deg=None,
        chunk_size=2,
        output_csv=output_csv,
    )
    monkeypatch.setattr(scan, "parse_args", lambda: args)

    scan.main()

    with output_csv.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))

    assert np.isclose(float(row["j_pixel_ref"]), 100.0)
    assert np.isclose(float(row["max_final_combined_pixel"]), 200.0)
    assert np.isclose(
        float(row["max_discarded_combined_theta_s_envelope_pixel"]),
        60.0,
    )
    assert np.isclose(
        float(
            row[
                "ratio_max_discarded_theta_s_envelope_to_j_pixel_ref"
            ]
        ),
        0.6,
    )


def test_dark_population_is_selected_before_reference_and_maps(
    monkeypatch,
    tmp_path,
):
    input_h5 = tmp_path / "dark_synthetic.h5"
    output_csv = tmp_path / "dark_scan.csv"
    column_names = [
        "Js",
        "D_Earth",
        "Vmax",
        "theta_s",
        "r_s",
        "rho_s",
        "Xearth",
        "Yearth",
        "Zearth",
    ]
    data = np.array(
        [
            # Bright dwarf: excluded before it can define J_ref.
            [1000.0, 10.0, 9.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            # Inclusive boundary: retained and defines dark J_ref.
            [100.0, 10.0, 8.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            # Retained in the uncut reference, discarded by f=0.5.
            [30.0, 10.0, 4.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0],
            [30.0, 10.0, 4.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    with h5py.File(input_h5, "w") as h5:
        group = h5.create_group("iteration_0")
        dataset = group.create_dataset("data", data=data)
        dataset.attrs["column_names"] = column_names

    def run(output, max_vmax):
        args = argparse.Namespace(
            repop_id=0,
            scenario="fragile",
            input_h5=input_h5,
            nside=1,
            pointlike_f_values="0.5",
            extended_f_values="0.5",
            theta_aperture_deg=None,
            chunk_size=2,
            output_csv=output,
            max_vmax_km_s=max_vmax,
        )
        monkeypatch.setattr(scan, "parse_args", lambda: args)
        scan.main()
        with output.open(newline="", encoding="utf-8") as stream:
            return next(csv.DictReader(stream))

    dark = run(output_csv, 8.0)
    assert dark["population_selection"] == "dark_vmax_le_8kms"
    assert float(dark["max_vmax_km_s"]) == 8.0
    assert float(dark["j_pixel_ref"]) == 100.0
    assert int(dark["n_physical_valid_total"]) == 4
    assert int(dark["n_vmax_excluded"]) == 1
    assert int(dark["n_valid_total"]) == 3
    assert int(dark["n_pointlike_total"]) == 3
    assert int(dark["n_pointlike_kept"]) == 1
    assert int(dark["n_pointlike_discarded"]) == 2
    # The reference includes both cut-discarded dark objects: 100+30+30.
    assert float(dark["pointlike_full_js_sum"]) == 160.0
    # The cut representation contains only the boundary object.
    assert float(dark["max_final_combined_pixel"]) == 100.0
    assert float(dark["max_discarded_pointlike_pixel"]) == 60.0
    assert float(
        dark["ratio_max_discarded_theta_s_envelope_to_j_pixel_ref"]
    ) == pytest.approx(0.6)

    full = run(tmp_path / "full_scan.csv", None)
    assert full["population_selection"] == "full"
    assert full["max_vmax_km_s"] == ""
    assert float(full["j_pixel_ref"]) == 1000.0
    assert int(full["n_vmax_excluded"]) == 0
    assert int(full["n_valid_total"]) == 4
