import importlib.util
from pathlib import Path

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
