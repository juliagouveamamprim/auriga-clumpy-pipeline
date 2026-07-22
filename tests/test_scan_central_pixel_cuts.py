import importlib.util
from pathlib import Path

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


def test_parse_theta_s_envelope():
    assert scan.parse_envelope_modes("theta-s") == ["theta-s"]
    assert scan.parse_envelope_modes("none") == []


def test_aperture_envelope_is_rejected():
    with pytest.raises(ValueError):
        scan.parse_envelope_modes("aperture")
