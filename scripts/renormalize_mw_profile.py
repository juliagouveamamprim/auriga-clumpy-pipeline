#!/usr/bin/env python3
"""Compute deterministic smooth-Milky-Way renormalization values.

This command reports the expected MHD subhalo mass and fixed-rs smooth-host
rescaling. It does not modify CLUMPY templates, configurations, maps, or
other scientific outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from renormalization import (  # noqa: E402
    integrate_expected_subhalo_mass,
    parameters_from_repository_config,
    renormalize_host,
    result_as_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("fragile", "resilient", "both"))
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "input_hydro.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON summary path; no file is written when omitted.",
    )
    parser.add_argument("--n-gauss-hermite", type=int, default=80)
    parser.add_argument("--n-vmax", type=int, default=16_385)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        input_data = yaml.safe_load(handle)
    scenarios = ("fragile", "resilient") if args.scenario == "both" else (args.scenario,)
    results: dict[str, dict] = {}
    for scenario in scenarios:
        parameters, host = parameters_from_repository_config(input_data, scenario)
        integral = integrate_expected_subhalo_mass(
            parameters,
            n_gauss_hermite=args.n_gauss_hermite,
            n_vmax=args.n_vmax,
        )
        host_result = renormalize_host(host, integral.m_sub_total)
        results[scenario] = result_as_dict(parameters, integral, host_result)
        print(
            f"{scenario}: N_expected={integral.n_expected:.8g}, "
            f"M_sub_total={integral.m_sub_total:.10e} Msun, "
            f"m_sub={host_result.m_sub_fraction:.6%}, "
            f"rho0_new/rho0_old={host_result.rho0_rescale_factor:.8f}, "
            f"gMW_RHOSOL_new={host_result.gmw_rhosol_new_gev_cm3:.10f} GeV/cm^3"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
