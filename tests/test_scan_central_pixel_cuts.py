import argparse
import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np


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
