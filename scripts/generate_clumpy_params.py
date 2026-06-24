#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate one CLUMPY parameter file from a renormalized template.

Usage
-----

Raw CLUMPY run:

    python3 generate_clumpy_params.py 1 fragile raw
    python3 generate_clumpy_params.py 1 resilient raw

Corrected CLUMPY run:

    python3 generate_clumpy_params.py 1 fragile corrected
    python3 generate_clumpy_params.py 1 resilient corrected

This script reads the renormalized template:

    configs/clumpy_templates/
        clumpy_params_g6_auriga_nfw_<scenario>_renorm_vmin0p1.template.txt

and generates:

    raw:
        outputs/clumpy/<scenario>/params/generated/
            repop_XXXX_raw_params.txt

    corrected:
        outputs/clumpy/<scenario>/params/generated/
            repop_XXXX_corrected_params.txt

It replaces, inside the template:

    gLIST_HALOES
    gSIM_OUTPUT_DIR

The physical MW normalization gMW_RHOSOL is already fixed in the
renormalized templates and is not modified here.
"""

import argparse
from pathlib import Path


# ============================================================
# GLOBAL CONFIG
# ============================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_RUN_DIR = REPOSITORY_ROOT / "outputs" / "clumpy"
TEMPLATE_DIR = REPOSITORY_ROOT / "configs" / "clumpy_templates"

TEMPLATE_NAME = (
    "clumpy_params_g6_auriga_nfw_{scenario}_renorm_vmin0p1.template.txt"
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one CLUMPY parameter file from a renormalized template."
    )

    parser.add_argument(
        "repop_id",
        type=int,
        help="Global repopulation ID. Example: 1 means repop_0001.",
    )

    parser.add_argument(
        "scenario",
        choices=["resilient", "fragile"],
        help="Hydro scenario.",
    )

    parser.add_argument(
        "stage",
        choices=["raw", "corrected"],
        help="CLUMPY run stage.",
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def replace_clumpy_line(line, key, new_value):
    """
    Replace the value field in a CLUMPY parameter line.

    Expected CLUMPY format:

        key   [unit]   value   <format>   comment...

    This keeps the key, unit, format and comment structure readable.
    """

    stripped = line.strip()

    if not stripped.startswith(key):
        return line, False

    parts = line.split()

    if len(parts) < 4:
        raise ValueError(f"Could not parse CLUMPY line:\n{line}")

    param_name = parts[0]
    unit = parts[1]

    # Everything after the old value is preserved.
    rest = " ".join(parts[3:])

    new_line = f"{param_name:<40} {unit:<14} {new_value} {rest}"

    return new_line, True


def get_paths(repop_id, scenario, stage):
    """
    Build all paths for one scenario/repop/stage.
    """

    if stage not in ("raw", "corrected"):
        raise ValueError("stage must be 'raw' or 'corrected'.")

    repop_tag = f"repop_{repop_id:04d}"
    scenario_dir = BASE_RUN_DIR / scenario

    template_path = (
        TEMPLATE_DIR
        / TEMPLATE_NAME.format(scenario=scenario)
    )

    if stage == "raw":
        input_list = (
            scenario_dir
            / "lists"
            / "raw"
            / f"{repop_tag}_raw_nopointlike.txt"
        )

        output_dir = (
            scenario_dir
            / "outputs"
            / "raw_clumpy"
            / repop_tag
        )

        output_param = (
            scenario_dir
            / "params"
            / "generated"
            / f"{repop_tag}_raw_params.txt"
        )

    else:
        input_list = (
            scenario_dir
            / "lists"
            / "corrected"
            / f"{repop_tag}_rhocorr.txt"
        )

        output_dir = (
            scenario_dir
            / "outputs"
            / "corrected_clumpy"
            / repop_tag
        )

        output_param = (
            scenario_dir
            / "params"
            / "generated"
            / f"{repop_tag}_corrected_params.txt"
        )

    return {
        "repop_tag": repop_tag,
        "scenario_dir": scenario_dir,
        "template_path": template_path,
        "input_list": input_list,
        "output_dir": output_dir,
        "output_param": output_param,
    }


def generate_one_param_file(repop_id, scenario, stage):
    """
    Generate one CLUMPY parameter file for one scenario/repop/stage.
    """

    paths = get_paths(
        repop_id=repop_id,
        scenario=scenario,
        stage=stage,
    )

    repop_tag = paths["repop_tag"]
    template_path = paths["template_path"]
    input_list = paths["input_list"]
    output_dir = paths["output_dir"]
    output_param = paths["output_param"]

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    if not input_list.exists():
        raise FileNotFoundError(f"Input CLUMPY list not found: {input_list}")

    output_param.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = template_path.read_text(encoding="utf-8").splitlines()

    new_lines = []

    header = [
        "# =============================================================================",
        "# Generated CLUMPY parameter file",
        "# =============================================================================",
        f"# Scenario: {scenario}",
        f"# REPOP_ID: {repop_id}",
        f"# REPOP_TAG: {repop_tag}",
        f"# Stage: {stage}",
        f"# Template: {template_path}",
        f"# Input halo list: {input_list}",
        f"# Output directory: {output_dir}",
        "#",
        "# This file was generated automatically from the renormalized Vmin=0.1 template.",
        "# Do not edit this generated file by hand unless debugging.",
        "# =============================================================================",
        "",
    ]

    found_list = False
    found_output = False

    for line in lines:
        line, replaced = replace_clumpy_line(
            line=line,
            key="gLIST_HALOES",
            new_value=str(input_list),
        )
        if replaced:
            found_list = True

        line, replaced = replace_clumpy_line(
            line=line,
            key="gSIM_OUTPUT_DIR",
            new_value=str(output_dir),
        )
        if replaced:
            found_output = True

        new_lines.append(line)

    if not found_list:
        raise RuntimeError(f"Did not find gLIST_HALOES in template: {template_path}")

    if not found_output:
        raise RuntimeError(f"Did not find gSIM_OUTPUT_DIR in template: {template_path}")

    output_param.write_text("\n".join(header + new_lines) + "\n", encoding="utf-8")

    print()
    print("=" * 80)
    print("Generated CLUMPY parameter file")
    print("=" * 80)
    print(f"Scenario: {scenario}")
    print(f"REPOP_ID: {repop_id}")
    print(f"REPOP_TAG: {repop_tag}")
    print(f"Stage: {stage}")
    print(f"Generated: {output_param}")
    print(f"Input list: {input_list}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)


def main():
    args = parse_args()

    generate_one_param_file(
        repop_id=args.repop_id,
        scenario=args.scenario,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
