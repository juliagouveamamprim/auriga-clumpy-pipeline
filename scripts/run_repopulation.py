#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run one Auriga full repopulation and save it in a pipeline-friendly directory.

Run from the repository root:

    python3 scripts/run_repopulation.py 1

This creates:

    outputs/repop_0001/

Safety rule:
    If the output directory already exists, the script stops and refuses
    to overwrite it.
"""

import argparse
import sys
from pathlib import Path


# ============================================================
# USER CONFIG
# ============================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT_DIR = REPOSITORY_ROOT / "outputs"
INPUT_YAML = REPOSITORY_ROOT / "configs" / "input_hydro.yml"

# Import the local repopulation implementation independently of the
# directory from which this script is executed.
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from repop_algorithm import RepopAlgorithm, read_config_file  # type: ignore


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one Auriga full repopulation for the CLUMPY pipeline."
    )

    parser.add_argument(
        "repop_id",
        type=int,
        help=(
            "Global repopulation ID. Example: 1 creates "
            "outputs/repop_0001/ inside the repository."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    repop_id = args.repop_id

    if repop_id < 0:
        raise ValueError("repop_id must be a non-negative integer.")

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outdir = BASE_OUTPUT_DIR / f"repop_{repop_id:04d}"

    print()
    print("=" * 80)
    print("Auriga full repopulation pipeline run")
    print("=" * 80)
    print(f"REPOP_ID: {repop_id}")
    print(f"Input YAML: {INPUT_YAML}")
    print(f"Output directory: {outdir}")
    print("=" * 80)
    print()

    if outdir.exists():
        raise FileExistsError(
            f"Output directory already exists:\n"
            f"  {outdir}\n\n"
            "Refusing to overwrite an existing repopulation. "
            "Choose a new repop_id or manually inspect the existing directory."
        )

    input_file = read_config_file(INPUT_YAML)

    repop_cfg = input_file.get("repopulations", {})
    print("Repopulation settings from YAML:")
    print(f"  RangeMin: {repop_cfg.get('RangeMin')}")
    print(f"  RangeMax: {repop_cfg.get('RangeMax')}")
    print(f"  number_iterations: {repop_cfg.get('number_iterations')}")
    print(f"  save_full_repop: {repop_cfg.get('save_full_repop')}")
    print(f"  rng_seed: {repop_cfg.get('rng_seed')}")
    print()

    model = RepopAlgorithm(input_file)

    model.run(str(outdir))

    print()
    print("=" * 80)
    print("Finished Auriga full repopulation")
    print("=" * 80)
    print(f"Output directory: {outdir}")
    print("Expected files:")
    print(f"  {outdir / 'input_data.yml'}")
    print(f"  {outdir / 'fullrepop_hydro_resilient.h5'}")
    print(f"  {outdir / 'fullrepop_hydro_fragile.h5'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
