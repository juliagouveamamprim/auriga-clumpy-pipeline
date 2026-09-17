"""Deterministic smooth-halo renormalization from the MHD subhalo population.

Numerical inputs use kpc, km/s, Msun, and Msun/kpc^3.  The configuration
adapter reads the unit-bearing repository YAML and extracts its values in
those units.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


CV_POLYNOMIAL = np.array([-0.90368, 0.2749, -0.028])
RMAX_OVER_RS = 2.163
DEFAULT_GMW_RHOSOL_OLD_GEV_CM3 = 0.3976794742


@dataclass(frozen=True)
class ScenarioParameters:
    """Scenario inputs; H0 is km/s/Mpc and G is kpc (km/s)^2/Msun."""

    scenario: str
    vmin: float
    vmax: float
    shvf_bb: float
    shvf_mm: float
    cv_c0: float
    sigma_log10_cv: float
    G: float
    H0: float


@dataclass(frozen=True)
class HostParameters:
    """Smooth-NFW host inputs: radii [kpc], rho0 [Msun/kpc^3]."""

    r_match: float
    rs: float
    rho0: float
    gmw_rhosol_old_gev_cm3: float = DEFAULT_GMW_RHOSOL_OLD_GEV_CM3


@dataclass(frozen=True)
class MassIntegral:
    """Deterministic SHVF integral plus resolved Vmax contributions."""

    n_expected: float
    m_sub_total: float
    log10_vmax: np.ndarray
    vmax: np.ndarray
    mean_subhalo_mass: np.ndarray
    differential_mass_per_log10_vmax: np.ndarray
    cumulative_mass: np.ndarray


@dataclass(frozen=True)
class HostRenormalization:
    """Mass-conserving NFW host rescaling with fixed rs and r_match."""

    m_smooth_old: float
    m_smooth_new: float
    m_sub_fraction: float
    rho0_host_old: float
    rho0_host_new: float
    rho0_rescale_factor: float
    gmw_rhosol_old_gev_cm3: float
    gmw_rhosol_new_gev_cm3: float


def f_nfw(x: np.ndarray | float) -> np.ndarray:
    """Return ln(1+x) - x/(1+x), the dimensionless NFW mass function."""
    x = np.asarray(x, dtype=float)
    return np.log1p(x) - x / (1.0 + x)


def nfw_mass_enclosed(
    radius_kpc: np.ndarray | float,
    rs_kpc: np.ndarray | float,
    rho0_msun_kpc3: np.ndarray | float,
) -> np.ndarray:
    """Return NFW M(<r) [Msun] for radii in kpc and rho0 in Msun/kpc^3."""
    radius_kpc = np.asarray(radius_kpc, dtype=float)
    rs_kpc = np.asarray(rs_kpc, dtype=float)
    rho0_msun_kpc3 = np.asarray(rho0_msun_kpc3, dtype=float)
    return 4.0 * np.pi * rho0_msun_kpc3 * rs_kpc**3 * f_nfw(radius_kpc / rs_kpc)


def cv_median(vmax_km_s: np.ndarray | float, c0: float) -> np.ndarray:
    """Return the Moliné+21 median velocity concentration cV(Vmax)."""
    y = np.log10(np.asarray(vmax_km_s, dtype=float))
    c1, c2, c3 = CV_POLYNOMIAL
    return c0 * (1.0 + c1 * y + c2 * y**2 + c3 * y**3)


def vmax_cv_to_nfw(
    vmax_km_s: np.ndarray | float,
    cv: np.ndarray | float,
    G_kpc_km2_s2_msun: float,
    H0_km_s_mpc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Vmax/cV to (rs [kpc], rho0 [Msun/kpc^3]).

    The 1e3 factor converts the Rmax expression from Mpc to kpc.
    """
    vmax_km_s = np.asarray(vmax_km_s, dtype=float)
    cv = np.asarray(cv, dtype=float)
    if np.any(vmax_km_s <= 0.0) or np.any(cv <= 0.0):
        raise ValueError("vmax and cV must be positive")
    rmax_kpc = vmax_km_s / H0_km_s_mpc * np.sqrt(2.0 / cv) * 1.0e3
    rs_kpc = rmax_kpc / RMAX_OVER_RS
    rho0_msun_kpc3 = (
        vmax_km_s**2
        * RMAX_OVER_RS
        / (4.0 * np.pi * G_kpc_km2_s2_msun * rs_kpc**2 * f_nfw(RMAX_OVER_RS))
    )
    return rs_kpc, rho0_msun_kpc3


def nfw_mass_within_rs(
    vmax_km_s: np.ndarray | float,
    cv: np.ndarray | float,
    G_kpc_km2_s2_msun: float,
    H0_km_s_mpc: float,
) -> np.ndarray:
    """Return M_sub(<rs) [Msun] from Vmax [km/s] and cV."""
    rs_kpc, rho0_msun_kpc3 = vmax_cv_to_nfw(
        vmax_km_s, cv, G_kpc_km2_s2_msun, H0_km_s_mpc
    )
    return nfw_mass_enclosed(rs_kpc, rs_kpc, rho0_msun_kpc3)


def shvf_expected_number(parameters: ScenarioParameters) -> float:
    """Analytically integrate dN/dV = 10**bb V**mm over the Vmax range."""
    exponent = parameters.shvf_mm + 1.0
    normalization = 10.0**parameters.shvf_bb
    if np.isclose(exponent, 0.0):
        return float(normalization * np.log(parameters.vmax / parameters.vmin))
    return float(normalization / exponent * (
        parameters.vmax**exponent - parameters.vmin**exponent
    ))


def mean_subhalo_mass_over_cv(
    vmax_km_s: np.ndarray | float,
    parameters: ScenarioParameters,
    n_gauss_hermite: int = 80,
) -> np.ndarray:
    """Return E[M_sub(<rs)|Vmax] over Normal(log10(cV))."""
    if n_gauss_hermite < 2:
        raise ValueError("n_gauss_hermite must be at least 2")
    vmax_km_s = np.asarray(vmax_km_s, dtype=float)
    cmed = cv_median(vmax_km_s, parameters.cv_c0)
    if np.any(~np.isfinite(cmed)) or np.any(cmed <= 0.0):
        raise ValueError("median cV must be finite and positive")
    nodes, weights = np.polynomial.hermite.hermgauss(n_gauss_hermite)
    log10_cv = np.log10(cmed)[..., None] + np.sqrt(2.0) * parameters.sigma_log10_cv * nodes
    masses = nfw_mass_within_rs(
        vmax_km_s[..., None], 10.0**log10_cv, parameters.G, parameters.H0
    )
    return np.sum(masses * weights / np.sqrt(np.pi), axis=-1)


def _cumulative_trapezoid(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(coordinate)
    return np.concatenate(([0.0], np.cumsum(increments)))


def integrate_expected_subhalo_mass(
    parameters: ScenarioParameters,
    n_gauss_hermite: int = 80,
    n_vmax: int = 16_385,
) -> MassIntegral:
    """Integrate dN/dV * E[M_sub(<rs)|V] on a log10(Vmax) grid."""
    if n_vmax < 3:
        raise ValueError("n_vmax must be at least 3")
    if not parameters.vmin > 0.0 or not parameters.vmax > parameters.vmin:
        raise ValueError("Vmax bounds must satisfy 0 < vmin < vmax")
    log10_vmax = np.linspace(np.log10(parameters.vmin), np.log10(parameters.vmax), n_vmax)
    vmax = 10.0**log10_vmax
    mean_mass = mean_subhalo_mass_over_cv(vmax, parameters, n_gauss_hermite)
    dndv = 10.0**parameters.shvf_bb * vmax**parameters.shvf_mm
    differential = np.log(10.0) * vmax * dndv * mean_mass
    cumulative = _cumulative_trapezoid(differential, log10_vmax)
    return MassIntegral(
        n_expected=shvf_expected_number(parameters),
        m_sub_total=float(cumulative[-1]),
        log10_vmax=log10_vmax,
        vmax=vmax,
        mean_subhalo_mass=mean_mass,
        differential_mass_per_log10_vmax=differential,
        cumulative_mass=cumulative,
    )


def renormalize_host(host: HostParameters, m_sub_total_msun: float) -> HostRenormalization:
    """Rescale host rho0 while keeping rs and r_match fixed."""
    if not np.isfinite(m_sub_total_msun) or m_sub_total_msun < 0.0:
        raise ValueError("m_sub_total_msun must be finite and non-negative")
    m_smooth_old = float(nfw_mass_enclosed(host.r_match, host.rs, host.rho0))
    m_sub_fraction = m_sub_total_msun / m_smooth_old
    if m_sub_fraction >= 1.0:
        raise ValueError("subhalo mass exceeds the old smooth-host mass")
    factor = 1.0 - m_sub_fraction
    rho0_host_new = host.rho0 * factor
    return HostRenormalization(
        m_smooth_old=m_smooth_old,
        m_smooth_new=float(nfw_mass_enclosed(host.r_match, host.rs, rho0_host_new)),
        m_sub_fraction=m_sub_fraction,
        rho0_host_old=host.rho0,
        rho0_host_new=rho0_host_new,
        rho0_rescale_factor=factor,
        gmw_rhosol_old_gev_cm3=host.gmw_rhosol_old_gev_cm3,
        gmw_rhosol_new_gev_cm3=host.gmw_rhosol_old_gev_cm3 * factor,
    )


def _config_value(value: Any) -> Any:
    return value["value"] if isinstance(value, dict) and "value" in value else value


def parameters_from_repository_config(
    input_data: dict[str, Any], scenario: str
) -> tuple[ScenarioParameters, HostParameters]:
    """Build inputs from configs/input_hydro.yml for fragile or resilient."""
    if scenario not in {"fragile", "resilient"}:
        raise ValueError("scenario must be 'fragile' or 'resilient'")
    configuration = input_data["configurations"][f"hydro_{scenario}"]
    shvf = configuration["SHVF"]["params"]
    cv = configuration["Cv"]["params"]
    cosmology = input_data["cosmo_constants"]
    repopulations = input_data["repopulations"]
    host_data = input_data["host"]
    parameters = ScenarioParameters(
        scenario=scenario,
        vmin=float(_config_value(repopulations["RangeMin"])),
        vmax=float(_config_value(repopulations["RangeMax"])),
        shvf_bb=float(_config_value(shvf["V0"])),
        shvf_mm=float(_config_value(shvf["slope"])),
        cv_c0=float(_config_value(cv["c0"])),
        sigma_log10_cv=float(_config_value(cv["sigma_scatter"])),
        G=float(_config_value(cosmology["G"])),
        H0=float(_config_value(cosmology["H_0"])),
    )
    host = HostParameters(
        r_match=float(_config_value(host_data["R_vir"])),
        rs=float(_config_value(host_data["r_s"])),
        rho0=float(_config_value(host_data["rho_0"])),
    )
    return parameters, host


def result_as_dict(
    parameters: ScenarioParameters,
    integral: MassIntegral,
    renormalization: HostRenormalization,
) -> dict[str, Any]:
    """Return scalar official results suitable for JSON serialization."""
    return {
        "parameters": asdict(parameters),
        "n_expected": integral.n_expected,
        "m_sub_total_msun": integral.m_sub_total,
        "m_sub_fraction": renormalization.m_sub_fraction,
        "rho0_rescale_factor": renormalization.rho0_rescale_factor,
        "gMW_RHOSOL_new_GeV_cm3": renormalization.gmw_rhosol_new_gev_cm3,
        "m_smooth_old_msun": renormalization.m_smooth_old,
        "m_smooth_new_msun": renormalization.m_smooth_new,
        "rho0_host_old_msun_kpc3": renormalization.rho0_host_old,
        "rho0_host_new_msun_kpc3": renormalization.rho0_host_new,
    }
