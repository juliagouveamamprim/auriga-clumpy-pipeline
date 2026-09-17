"""Validate deterministic smooth-halo normalizations in CLUMPY templates."""

from pathlib import Path
import re

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "fragile": (
        REPOSITORY_ROOT
        / "configs/clumpy_templates/clumpy_params_g6_auriga_nfw_fragile_renorm_vmin0p1.template.txt",
        0.3949787768,
    ),
    "resilient": (
        REPOSITORY_ROOT
        / "configs/clumpy_templates/clumpy_params_g6_auriga_nfw_resilient_renorm_vmin0p1.template.txt",
        0.3932994559,
    ),
}


@pytest.mark.parametrize("scenario", ("fragile", "resilient"))
def test_template_uses_approved_deterministic_rhosol(scenario):
    path, expected_rhosol = TEMPLATES[scenario]
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^gMW_RHOSOL\s+\[GeV/cm3\]\s+([^\s]+)", text, flags=re.MULTILINE)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(expected_rhosol, rel=0.0, abs=1e-12)
    assert "expectation integrated deterministically" in text
    assert "Gauss-Hermite" in text
    assert "M_sub = M(<r_s)" in text
    assert "Catalogue cuts do not enter" in text
    for obsolete_mc_text in ("N_MC", "seed:", "Monte Carlo", "mean individual M_sub"):
        assert obsolete_mc_text not in text
