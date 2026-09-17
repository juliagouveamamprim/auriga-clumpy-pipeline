"""Tests for the official deterministic smooth-halo renormalization."""

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from renormalization import (  # noqa: E402
    HostParameters,
    cv_median,
    integrate_expected_subhalo_mass,
    mean_subhalo_mass_over_cv,
    nfw_mass_within_rs,
    parameters_from_repository_config,
    renormalize_host,
    vmax_cv_to_nfw,
)


REFERENCE = {
    "fragile": (77884185.65689036, 9.67884547e9, 0.00679114, 0.99320886, 0.3949787768),
    "resilient": (126313358.72292233, 1.56972493e10, 0.01101394, 0.98898606, 0.3932994559),
}


@pytest.fixture(scope="module")
def input_data():
    with (REPOSITORY_ROOT / "configs" / "input_hydro.yml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_nfw_conversion_preserves_kpc_mpc_unit_factor():
    rs, rho0 = vmax_cv_to_nfw(20.0, 1.0e4, 4.297e-6, 100.0)
    expected_rmax_kpc = 20.0 / 100.0 * np.sqrt(2.0 / 1.0e4) * 1.0e3
    np.testing.assert_allclose(rs, expected_rmax_kpc / 2.163, rtol=1e-14)
    assert rho0 > 0.0
    mass = nfw_mass_within_rs(20.0, 1.0e4, 4.297e-6, 100.0)
    assert np.isfinite(mass) and mass > 0.0


def test_concentration_and_conditional_mass_are_finite_and_positive(input_data):
    parameters, _ = parameters_from_repository_config(input_data, "fragile")
    vmax = np.geomspace(parameters.vmin, parameters.vmax, 101)
    assert np.all(cv_median(vmax, parameters.cv_c0) > 0.0)
    conditional_mass = mean_subhalo_mass_over_cv(vmax, parameters, n_gauss_hermite=20)
    assert np.all(np.isfinite(conditional_mass)) and np.all(conditional_mass > 0.0)


@pytest.mark.parametrize("scenario", ("fragile", "resilient"))
def test_references_and_host_mass_conservation(input_data, scenario):
    parameters, host = parameters_from_repository_config(input_data, scenario)
    integral = integrate_expected_subhalo_mass(parameters)
    result = renormalize_host(host, integral.m_sub_total)
    expected = REFERENCE[scenario]
    np.testing.assert_allclose(integral.n_expected, expected[0], rtol=5e-12)
    np.testing.assert_allclose(integral.m_sub_total, expected[1], rtol=5e-9)
    np.testing.assert_allclose(result.m_sub_fraction, expected[2], rtol=5e-7)
    np.testing.assert_allclose(result.rho0_rescale_factor, expected[3], rtol=5e-9)
    np.testing.assert_allclose(result.gmw_rhosol_new_gev_cm3, expected[4], rtol=5e-9)
    np.testing.assert_allclose(result.m_smooth_new + integral.m_sub_total, result.m_smooth_old, rtol=1e-14)


def test_gauss_hermite_quadrature_converges(input_data):
    parameters, _ = parameters_from_repository_config(input_data, "fragile")
    vmax = np.geomspace(parameters.vmin, parameters.vmax, 17)
    coarse = mean_subhalo_mass_over_cv(vmax, parameters, n_gauss_hermite=10)
    fine = mean_subhalo_mass_over_cv(vmax, parameters, n_gauss_hermite=80)
    np.testing.assert_allclose(coarse, fine, rtol=1e-12)


def test_vmax_resolution_converges(input_data):
    parameters, _ = parameters_from_repository_config(input_data, "fragile")
    coarse = integrate_expected_subhalo_mass(parameters, n_gauss_hermite=80, n_vmax=1025)
    fine = integrate_expected_subhalo_mass(parameters, n_gauss_hermite=80, n_vmax=16385)
    assert abs(coarse.m_sub_total / fine.m_sub_total - 1.0) < 5e-7


def test_unphysical_host_renormalization_is_rejected():
    host = HostParameters(r_match=220.0, rs=20.0, rho0=9.04e6)
    with pytest.raises(ValueError, match="exceeds"):
        renormalize_host(host, 2.0e12)
