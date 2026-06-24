#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correct CLUMPY halo-list rhos values using the J actually rendered by CLUMPY.

Usage
-----

    python3 correct_rhos_from_clumpy_raw.py 1 fragile
    python3 correct_rhos_from_clumpy_raw.py 1 resilient

This script reads:

    outputs/clumpy/<scenario>/lists/raw/
        repop_XXXX_raw_nopointlike.txt

and the corresponding CLUMPY raw rendered-halo log:

    outputs/clumpy/<scenario>/outputs/raw_clumpy/
        repop_XXXX/annihil_gal2D_LOS0_0_FOV360x180_nside1024.halo_rendered.log

It computes, halo by halo:

    R = J_rendered / J_expected

where J_expected is the analytic NFW J(<r_s) computed from the CLUMPY list.

The correction applied to the CLUMPY list is:

    X = 1 / sqrt(R)
    rhos_new = X * rhos_old

because J scales as rho_s^2.

The corrected list is written to:

    outputs/clumpy/<scenario>/lists/corrected/
        repop_XXXX_rhocorr.txt
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# GLOBAL CONFIG
# ============================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_RUN_DIR = REPOSITORY_ROOT / "outputs" / "clumpy"

# CLUMPY output basename currently produced by the g6 full-sky run.
CLUMPY_BASENAME = "annihil_gal2D_LOS0_0_FOV360x180_nside1024"

# Conversion factor:
# [Msun^2 / kpc^5] -> [GeV^2 / cm^5]
MSUN2_KPC5_TO_GEV2_CM5 = 4.446e6


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Correct CLUMPY halo-list rhos values from raw CLUMPY rendered logs."
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

    return parser.parse_args()


# ============================================================
# Path helpers
# ============================================================

def get_paths(repop_id, scenario):
    repop_tag = f"repop_{repop_id:04d}"
    scenario_dir = BASE_RUN_DIR / scenario

    input_list = (
        scenario_dir
        / "lists"
        / "raw"
        / f"{repop_tag}_raw_nopointlike.txt"
    )

    raw_clumpy_output_dir = (
        scenario_dir
        / "outputs"
        / "raw_clumpy"
        / repop_tag
    )

    rendered_log = raw_clumpy_output_dir / f"{CLUMPY_BASENAME}.halo_rendered.log"

    output_list = (
        scenario_dir
        / "lists"
        / "corrected"
        / f"{repop_tag}_rhocorr.txt"
    )

    output_dir = (
        scenario_dir
        / "logs"
        / "correction"
        / repop_tag
    )

    output_table = output_dir / f"matched_R_diagnostics_{scenario}_{repop_tag}.csv"

    return {
        "repop_tag": repop_tag,
        "scenario_dir": scenario_dir,
        "input_list": input_list,
        "raw_clumpy_output_dir": raw_clumpy_output_dir,
        "rendered_log": rendered_log,
        "output_list": output_list,
        "output_dir": output_dir,
        "output_table": output_table,
        "plot_R_vs_theta": output_dir / "R_vs_theta.png",
        "plot_X_vs_theta": output_dir / "X_vs_theta.png",
        "plot_R_hist": output_dir / "R_hist.png",
        "plot_R_vs_d": output_dir / "R_vs_distance.png",
        "plot_R_vs_npix": output_dir / "R_vs_npix.png",
    }


# ============================================================
# READERS
# ============================================================

def load_clumpy_list(path):
    """
    Read a CLUMPY halo list file.

    Expected format for non-comment lines:

        Name Type l b d z Rdelta rhos rs prof #1 #2 #3
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input CLUMPY list not found: {path}")

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()

            if (not s) or s.startswith("#"):
                continue

            parts = s.split()

            if len(parts) < 13:
                continue

            rows.append({
                "halo_name": parts[0],
                "type": parts[1],
                "l_deg": float(parts[2]),
                "b_deg": float(parts[3]),
                "d_kpc": float(parts[4]),
                "z": float(parts[5]),
                "Rdelta_kpc": float(parts[6]),
                "rhos_old": float(parts[7]),
                "rs_kpc": float(parts[8]),
                "prof": parts[9],
                "p1": int(float(parts[10])),
                "p2": int(float(parts[11])),
                "p3": int(float(parts[12])),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No halo rows found in the CLUMPY list file: {path}")

    return df


def load_rendered_log(path):
    """
    Read CLUMPY rendered-halo log with rows like:

        HALO_INDEX NAME TYPE J_RENDERED NPIX_TOUCHED

    The header line may be commented with '#', so we assign column names
    explicitly.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Rendered log not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["HALO_INDEX", "NAME", "TYPE", "J_RENDERED", "NPIX_TOUCHED"],
        engine="python",
    )

    df = df.rename(columns={
        "NAME": "halo_name",
        "TYPE": "rendered_type",
        "J_RENDERED": "J_rendered",
        "NPIX_TOUCHED": "npix_touched",
    })

    df["HALO_INDEX"] = pd.to_numeric(df["HALO_INDEX"], errors="coerce")
    df["J_rendered"] = pd.to_numeric(df["J_rendered"], errors="coerce")
    df["npix_touched"] = pd.to_numeric(df["npix_touched"], errors="coerce")

    df = df.dropna(subset=["halo_name", "J_rendered"]).copy()
    df["HALO_INDEX"] = df["HALO_INDEX"].astype(int)

    if df.empty:
        raise RuntimeError(f"No valid rows found in rendered log: {path}")

    return df


# ============================================================
# ANALYTIC J(<rs) FOR NFW
# ============================================================

def j_expected_from_list(df):
    """
    Compute analytic J(<rs) for NFW halos directly from the CLUMPY list.

    The list stores:

        rhos = rho(rs) = rho_s / 4

    For NFW integrated up to r_s:

        J(<r_s) = (7*pi/6) * rho_s^2 * r_s^3 / D^2

    Since rho_s = 4 * rhos:

        J(<r_s) = (56*pi/3) * rhos^2 * r_s^3 / D^2

    Units:
        rhos in Msun/kpc^3
        r_s and D in kpc
        J in Msun^2/kpc^5

    Then converted to GeV^2/cm^5.
    """
    out = df.copy()

    d = out["d_kpc"].to_numpy(dtype=float)
    rs = out["rs_kpc"].to_numpy(dtype=float)
    rhos = out["rhos_old"].to_numpy(dtype=float)

    j_msun2_kpc5 = (56.0 * math.pi / 3.0) * (rhos ** 2) * (rs ** 3) / (d ** 2)
    j_gev2_cm5 = j_msun2_kpc5 * MSUN2_KPC5_TO_GEV2_CM5

    out["J_expected"] = j_gev2_cm5

    # Angular size derived from list itself, for diagnostics.
    out["theta_deg_from_list"] = np.degrees(np.arctan(rs / d))

    # Recover standard NFW scale density too.
    out["rho_s_old"] = 4.0 * out["rhos_old"]

    return out


# ============================================================
# CORE
# ============================================================

def build_matched_table(list_df, rendered_df):
    """
    Match list halos with rendered halos by halo_name and compute:

        R = J_rendered / J_expected
        X = 1/sqrt(R)
        rhos_new = X * rhos_old
    """
    df = list_df.merge(
        rendered_df[["halo_name", "J_rendered", "npix_touched", "HALO_INDEX"]],
        on="halo_name",
        how="inner",
        validate="one_to_one",
    )

    valid = (
        np.isfinite(df["J_expected"]) & (df["J_expected"] > 0.0) &
        np.isfinite(df["J_rendered"]) & (df["J_rendered"] > 0.0) &
        np.isfinite(df["rhos_old"]) & (df["rhos_old"] > 0.0)
    )

    df = df[valid].copy()

    df["R"] = df["J_rendered"] / df["J_expected"]
    df["X"] = 1.0 / np.sqrt(df["R"])
    df["rhos_new"] = df["X"] * df["rhos_old"]
    df["rho_s_new"] = 4.0 * df["rhos_new"]

    return df


# ============================================================
# PLOTS
# ============================================================

def make_plots(df, paths):
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["theta_deg_from_list"], df["R"], s=10)
    plt.xscale("log")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Angular size [deg]")
    plt.ylabel("R = J_rendered / J_expected")
    plt.tight_layout()
    plt.savefig(paths["plot_R_vs_theta"], dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["theta_deg_from_list"], df["X"], s=10)
    plt.xscale("log")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Angular size [deg]")
    plt.ylabel("X = 1 / sqrt(R)")
    plt.tight_layout()
    plt.savefig(paths["plot_X_vs_theta"], dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(df["R"], bins=60)
    plt.axvline(1.0, linestyle="--")
    plt.xlabel("R = J_rendered / J_expected")
    plt.ylabel("Number of halos")
    plt.tight_layout()
    plt.savefig(paths["plot_R_hist"], dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["d_kpc"], df["R"], s=10)
    plt.xscale("log")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Distance [kpc]")
    plt.ylabel("R = J_rendered / J_expected")
    plt.tight_layout()
    plt.savefig(paths["plot_R_vs_d"], dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["npix_touched"], df["R"], s=10)
    plt.xscale("log")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("NPIX_TOUCHED")
    plt.ylabel("R = J_rendered / J_expected")
    plt.tight_layout()
    plt.savefig(paths["plot_R_vs_npix"], dpi=180)
    plt.close()


# ============================================================
# WRITER
# ============================================================

def write_corrected_list(
    original_list_path,
    output_list_path,
    corr_df,
    scenario,
    repop_id,
    paths,
):
    """
    Rewrite the original CLUMPY list, replacing only rhos for matched halos.

    Keeps:
        - same halo names
        - same ordering as the input list
        - same comments/header, plus an added correction note
    """
    original_list_path = Path(original_list_path)
    output_list_path = Path(output_list_path)
    output_list_path.parent.mkdir(parents=True, exist_ok=True)

    corr_map = corr_df.set_index("halo_name")["rhos_new"].to_dict()

    out_lines = []

    correction_header = [
        "#",
        "# =============================================================================",
        "# rho_s correction applied",
        "# =============================================================================",
        f"# Scenario: {scenario}",
        f"# REPOP_ID: {repop_id}",
        f"# Raw list: {paths['input_list']}",
        f"# Rendered log: {paths['rendered_log']}",
        f"# Correction table: {paths['output_table']}",
        "# Correction rule:",
        "#   R = J_rendered / J_expected",
        "#   X = 1 / sqrt(R)",
        "#   rhos_new = X * rhos_old",
        "# =============================================================================",
        "#",
    ]

    inserted_header = False

    with open(original_list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()

            if (not inserted_header) and s.startswith("# Format:"):
                out_lines.extend(correction_header)
                inserted_header = True

            if (not s) or s.startswith("#"):
                out_lines.append(line.rstrip("\n"))
                continue

            parts = s.split()

            if len(parts) < 13:
                out_lines.append(line.rstrip("\n"))
                continue

            halo_name = parts[0]

            if halo_name in corr_map:
                parts[7] = f"{corr_map[halo_name]:.8e}"

            new_line = (
                f"{parts[0]:<15s} "
                f"{parts[1]:<8s} "
                f"{float(parts[2]):>10.6f} "
                f"{float(parts[3]):>10.6f} "
                f"{float(parts[4]):>12.6f} "
                f"{float(parts[5]):>6.0f} "
                f"{float(parts[6]):>12.6f} "
                f"{parts[7]:>16s} "
                f"{float(parts[8]):>10.6f} "
                f"{parts[9]:<8s} "
                f"{int(float(parts[10])):>4d} "
                f"{int(float(parts[11])):>4d} "
                f"{int(float(parts[12])):>4d}"
            )

            out_lines.append(new_line)

    if not inserted_header:
        out_lines = correction_header + out_lines

    with open(output_list_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    repop_id = args.repop_id
    scenario = args.scenario

    paths = get_paths(repop_id=repop_id, scenario=scenario)

    print()
    print("=" * 80)
    print("CLUMPY rho_s correction")
    print("=" * 80)
    print(f"Scenario: {scenario}")
    print(f"REPOP_ID: {repop_id}")
    print(f"REPOP_TAG: {paths['repop_tag']}")
    print(f"Input list: {paths['input_list']}")
    print(f"Rendered log: {paths['rendered_log']}")
    print(f"Output corrected list: {paths['output_list']}")
    print(f"Diagnostics output dir: {paths['output_dir']}")
    print("=" * 80)
    print()

    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    list_df = load_clumpy_list(paths["input_list"])
    list_df = j_expected_from_list(list_df)

    rendered_df = load_rendered_log(paths["rendered_log"])

    matched = build_matched_table(list_df, rendered_df)

    if matched.empty:
        raise RuntimeError(
            "No matched valid halos found. Check halo names in list and rendered log."
        )

    matched.to_csv(paths["output_table"], index=False)

    make_plots(matched, paths)

    write_corrected_list(
        original_list_path=paths["input_list"],
        output_list_path=paths["output_list"],
        corr_df=matched,
        scenario=scenario,
        repop_id=repop_id,
        paths=paths,
    )

    n_list = len(list_df)
    n_log = len(rendered_df)
    n_matched = len(matched)

    print(f"Halos in input list: {n_list}")
    print(f"Halos in rendered log: {n_log}")
    print(f"Matched valid halos: {n_matched}")
    print()
    print(f"Median R: {np.nanmedian(matched['R']):.6f}")
    print(f"Mean R:   {np.nanmean(matched['R']):.6f}")
    print(f"Median X: {np.nanmedian(matched['X']):.6f}")
    print(f"Mean X:   {np.nanmean(matched['X']):.6f}")
    print()
    print(f"Saved diagnostics table: {paths['output_table']}")
    print(f"Saved plot: {paths['plot_R_vs_theta']}")
    print(f"Saved plot: {paths['plot_X_vs_theta']}")
    print(f"Saved plot: {paths['plot_R_hist']}")
    print(f"Saved plot: {paths['plot_R_vs_d']}")
    print(f"Saved plot: {paths['plot_R_vs_npix']}")
    print(f"Saved corrected list: {paths['output_list']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
