#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correct CLUMPY halo-list rhos values using the J actually rendered by CLUMPY.

Usage
-----

    python3 scripts/correct_rhos_from_clumpy_raw.py 1 fragile
    python3 scripts/correct_rhos_from_clumpy_raw.py 1 resilient

This script reads:

    outputs/clumpy/<scenario>/lists/raw/
        repop_XXXX_raw_nopointlike_nside<NSIDE>.txt

and the corresponding CLUMPY raw rendered-halo log:

    outputs/clumpy/<scenario>/outputs/raw_clumpy/
        repop_XXXX_nside<NSIDE>/annihil_gal2D_LOS0_0_FOV360x180_nside<NSIDE>.halo_rendered.log

It computes, halo by halo:

    R_raw = J_raw / J_catalog

where J_catalog is read directly from the authoritative HDF5 catalogue by
exact halo name.

The correction applied to the CLUMPY list is:

    correction_factor = sqrt(J_catalog / J_raw)
    rhos_corrected = correction_factor * rhos_raw

because J scales as rho_s^2.

The corrected list is written to:

    outputs/clumpy/<scenario>/lists/corrected/
        repop_XXXX_rhocorr_nside<NSIDE>.txt
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u


# ============================================================
# GLOBAL CONFIG
# ============================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_RUN_DIR = REPOSITORY_ROOT / "outputs" / "clumpy"

DEFAULT_NSIDE = 2048
DEFAULT_CATALOG_ROOT = REPOSITORY_ROOT / "outputs"
HALO_NAME_PATTERN = re.compile(r"^hydro_res_(0*[1-9][0-9]*)$")


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

    parser.add_argument(
        "--nside",
        type=int,
        default=DEFAULT_NSIDE,
        help=(
            "HEALPix NSIDE of the raw CLUMPY run. "
            "Output filenames always include the NSIDE value."
        ),
    )

    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=DEFAULT_CATALOG_ROOT,
        help=(
            "Root containing repop_XXXX/fullrepop_hydro_<scenario>.h5. "
            "This may be separate from the CLUMPY product root."
        ),
    )

    parser.add_argument(
        "--clumpy-root",
        type=Path,
        default=BASE_RUN_DIR,
        help="Root containing the scenario-specific CLUMPY products.",
    )

    parser.add_argument("--raw-list", type=Path, default=None)
    parser.add_argument("--raw-rendered-log", type=Path, default=None)
    parser.add_argument("--output-list", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--metadata-output-dir",
        type=Path,
        default=None,
        help=(
            "Final promoted correction directory to record in metadata when "
            "writing into a staging directory."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the raw list/log and authoritative HDF5 matching "
            "without writing correction products."
        ),
    )

    return parser.parse_args()


# ============================================================
# Path helpers
# ============================================================

def nside_suffix(nside):
    return f"_nside{nside}"


def clumpy_repop_dir(repop_tag, nside):
    return f"{repop_tag}{nside_suffix(nside)}"


def clumpy_basename(nside):
    return f"annihil_gal2D_LOS0_0_FOV360x180_nside{nside}"


def get_paths(
    repop_id,
    scenario,
    nside,
    catalog_root=None,
    clumpy_root=None,
):
    repop_tag = f"repop_{repop_id:04d}"
    if catalog_root is None:
        catalog_root = DEFAULT_CATALOG_ROOT
    if clumpy_root is None:
        clumpy_root = BASE_RUN_DIR
    catalog_root = Path(catalog_root)
    scenario_dir = Path(clumpy_root) / scenario
    suffix = nside_suffix(nside)
    output_repop_dir = clumpy_repop_dir(repop_tag, nside)

    input_list = (
        scenario_dir
        / "lists"
        / "raw"
        / f"{repop_tag}_raw_nopointlike{suffix}.txt"
    )

    raw_clumpy_output_dir = (
        scenario_dir
        / "outputs"
        / "raw_clumpy"
        / output_repop_dir
    )

    rendered_log = raw_clumpy_output_dir / f"{clumpy_basename(nside)}.halo_rendered.log"

    output_list = (
        scenario_dir
        / "lists"
        / "corrected"
        / f"{repop_tag}_rhocorr{suffix}.txt"
    )

    output_dir = (
        scenario_dir
        / "logs"
        / "correction"
        / output_repop_dir
    )

    output_table = output_dir / f"matched_R_diagnostics_{scenario}_{repop_tag}{suffix}.csv"

    return {
        "repop_tag": repop_tag,
        "scenario_dir": scenario_dir,
        "catalog": (
            catalog_root
            / repop_tag
            / f"fullrepop_hydro_{scenario}.h5"
        ),
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

            if len(parts) != 13:
                raise RuntimeError(
                    f"Malformed CLUMPY-list row in {path}: {s!r}"
                )

            rows.append({
                "halo_name": parts[0],
                "type": parts[1],
                "l_deg": float(parts[2]),
                "b_deg": float(parts[3]),
                "d_kpc": float(parts[4]),
                "z": float(parts[5]),
                "Rdelta_kpc": float(parts[6]),
                "rhos_raw": float(parts[7]),
                "rs_kpc": float(parts[8]),
                "prof": parts[9],
                "p1": int(float(parts[10])),
                "p2": int(float(parts[11])),
                "p3": int(float(parts[12])),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No halo rows found in the CLUMPY list file: {path}")

    duplicate_names = df.loc[
        df["halo_name"].duplicated(keep=False), "halo_name"
    ].unique()
    if duplicate_names.size:
        raise RuntimeError(
            "Raw CLUMPY list contains duplicate halo names: "
            + ", ".join(map(str, duplicate_names[:5]))
        )

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

    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise RuntimeError(
                    f"Malformed rendered-log row {line_number} in {path}: "
                    f"expected 5 fields, found {len(parts)}"
                )
            try:
                halo_index = int(parts[0])
                j_raw = float(parts[3])
                npix_touched = int(parts[4])
            except ValueError as error:
                raise RuntimeError(
                    f"Malformed rendered-log row {line_number} in {path}: "
                    f"{stripped!r}"
                ) from error
            rows.append(
                {
                    "HALO_INDEX": halo_index,
                    "halo_name": parts[1],
                    "rendered_type": parts[2],
                    "J_raw": j_raw,
                    "npix_touched": npix_touched,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No valid rows found in rendered log: {path}")

    duplicate_names = df.loc[
        df["halo_name"].duplicated(keep=False), "halo_name"
    ].unique()
    if duplicate_names.size:
        raise RuntimeError(
            "Raw rendered log contains duplicate halo names: "
            + ", ".join(map(str, duplicate_names[:5]))
        )

    expected_indices = np.arange(len(df), dtype=np.int64)
    actual_indices = df["HALO_INDEX"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_indices, expected_indices):
        raise RuntimeError(
            "Raw rendered-log HALO_INDEX must be exactly zero-based 0..N-1"
        )

    return df


# ============================================================
# TARGETED HDF5 CATALOGUE LOOKUP
# ============================================================

def _decode_strings(values):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _validate_js_unit(unit_text, catalog_path):
    try:
        parsed = u.Unit(unit_text)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid Js unit {unit_text!r} in {catalog_path}"
        ) from error
    if not parsed.is_equivalent(u.GeV**2 / u.cm**5):
        raise RuntimeError(
            f"Unexpected Js unit {unit_text!r} in {catalog_path}; "
            "expected GeV^2 cm^-5"
        )


def load_catalog_js_indexed(catalog_path, target_names):
    """Read only requested catalogue Js rows and preserve list order."""
    catalog_path = Path(catalog_path)
    if not catalog_path.is_file() or catalog_path.stat().st_size == 0:
        raise FileNotFoundError(f"HDF5 catalogue not found or empty: {catalog_path}")

    if len(set(target_names)) != len(target_names):
        raise RuntimeError("Requested HDF5 halo names are not unique")

    try:
        with h5py.File(catalog_path, "r") as handle:
            if "iteration_0" not in handle:
                raise RuntimeError(
                    f"Missing HDF5 group iteration_0 in {catalog_path}"
                )
            group = handle["iteration_0"]
            if "data" not in group or "halo_name" not in group:
                raise RuntimeError(
                    f"Missing iteration_0/data or iteration_0/halo_name in "
                    f"{catalog_path}"
                )
            data = group["data"]
            stored_names = group["halo_name"]
            if data.ndim != 2:
                raise RuntimeError(
                    f"iteration_0/data must be two-dimensional in {catalog_path}"
                )
            if stored_names.shape[0] != data.shape[0]:
                raise RuntimeError(
                    f"halo_name length {stored_names.shape[0]} does not match "
                    f"data length {data.shape[0]} in {catalog_path}"
                )
            n_saved = group.attrs.get("n_saved")
            if n_saved is None or int(n_saved) != data.shape[0]:
                raise RuntimeError(
                    f"HDF5 n_saved metadata does not match data length in "
                    f"{catalog_path}"
                )
            if "column_names" not in data.attrs or "units" not in data.attrs:
                raise RuntimeError(
                    f"HDF5 data is missing column_names or units in {catalog_path}"
                )
            column_names = _decode_strings(data.attrs["column_names"])
            units = _decode_strings(data.attrs["units"])
            if len(column_names) != data.shape[1] or len(units) != data.shape[1]:
                raise RuntimeError(
                    f"HDF5 column_names/units length mismatch in {catalog_path}"
                )
            if column_names.count("Js") != 1:
                raise RuntimeError(
                    f"HDF5 must contain exactly one Js column in {catalog_path}"
                )
            js_index = column_names.index("Js")
            _validate_js_unit(units[js_index], catalog_path)

            inferred_rows = []
            for name in target_names:
                match = HALO_NAME_PATTERN.fullmatch(name)
                if match is None:
                    raise RuntimeError(
                        f"Non-canonical halo name {name!r}; expected "
                        "hydro_res_<positive numeric suffix>"
                    )
                row_index = int(match.group(1)) - 1
                if row_index < 0 or row_index >= data.shape[0]:
                    raise RuntimeError(
                        f"Halo name {name!r} implies HDF5 row {row_index}, "
                        f"outside 0..{data.shape[0] - 1}"
                    )
                inferred_rows.append(row_index)

            if len(set(inferred_rows)) != len(inferred_rows):
                raise RuntimeError("Inferred HDF5 row indices are not unique")

            rows = np.asarray(inferred_rows, dtype=np.int64)
            sort_order = np.argsort(rows, kind="stable")
            sorted_rows = rows[sort_order]
            expected_sorted_names = [target_names[index] for index in sort_order]
            actual_sorted_names = _decode_strings(stored_names[sorted_rows])
            sorted_js = np.asarray(
                data[sorted_rows, js_index], dtype=np.float64
            )

            for row_index, expected_name, actual_name in zip(
                sorted_rows, expected_sorted_names, actual_sorted_names
            ):
                if actual_name != expected_name:
                    raise RuntimeError(
                        f"HDF5 identity mismatch at row {int(row_index)}: "
                        f"expected {expected_name!r}, found {actual_name!r}"
                    )
    except OSError as error:
        raise RuntimeError(
            f"Could not open HDF5 catalogue {catalog_path}: {error}"
        ) from error

    catalog_js = np.empty(len(sorted_js), dtype=np.float64)
    catalog_js[sort_order] = sorted_js
    if np.any(~np.isfinite(catalog_js)) or np.any(catalog_js <= 0.0):
        raise RuntimeError("Matched HDF5 J_catalog values must be finite and positive")
    return rows, catalog_js


# ============================================================
# CORE
# ============================================================

def validate_raw_correspondence(list_df, rendered_df):
    """Require a complete, ordered, one-to-one raw-list/log correspondence."""
    raw_names = set(list_df["halo_name"])
    rendered_names = set(rendered_df["halo_name"])
    raw_only = raw_names - rendered_names
    log_only = rendered_names - raw_names

    if raw_only or log_only:
        raise RuntimeError(
            "Halo identifier mismatch: "
            f"raw-only={len(raw_only)}, log-only={len(log_only)}"
        )
    if len(list_df) != len(rendered_df):
        raise RuntimeError(
            f"Raw list/log row-count mismatch: {len(list_df)} != "
            f"{len(rendered_df)}"
        )
    list_names = list_df["halo_name"].tolist()
    log_names = rendered_df["halo_name"].tolist()
    if log_names != list_names:
        raise RuntimeError(
            "Raw rendered-log names/order do not match the raw CLUMPY list"
        )
    if rendered_df["rendered_type"].tolist() != list_df["type"].tolist():
        raise RuntimeError(
            "Raw rendered-log TYPE sequence does not match the raw CLUMPY list"
        )


def build_matched_table(list_df, rendered_df, hdf5_rows, catalog_js):
    """
    Match list halos with rendered halos by halo_name and compute:

        R_raw = J_raw / J_catalog
        correction_factor = sqrt(J_catalog / J_raw)
        rhos_corrected = correction_factor * rhos_raw
    """
    validate_raw_correspondence(list_df, rendered_df)
    if len(hdf5_rows) != len(list_df) or len(catalog_js) != len(list_df):
        raise RuntimeError("HDF5 lookup result length does not match raw list")

    df = list_df.copy()
    df["HALO_INDEX"] = rendered_df["HALO_INDEX"].to_numpy(dtype=np.int64)
    df["hdf5_row"] = np.asarray(hdf5_rows, dtype=np.int64)
    df["J_catalog"] = np.asarray(catalog_js, dtype=np.float64)
    df["J_raw"] = rendered_df["J_raw"].to_numpy(dtype=np.float64)
    df["npix_touched"] = rendered_df["npix_touched"].to_numpy(dtype=np.int64)

    if np.any(~np.isfinite(df["J_raw"])) or np.any(df["J_raw"] <= 0.0):
        raise RuntimeError("J_raw values must be finite and positive")
    if np.any(~np.isfinite(df["J_catalog"])) or np.any(df["J_catalog"] <= 0.0):
        raise RuntimeError("J_catalog values must be finite and positive")
    if np.any(~np.isfinite(df["rhos_raw"])) or np.any(df["rhos_raw"] <= 0.0):
        raise RuntimeError("rhos_raw values must be finite and positive")

    df["R_raw"] = df["J_raw"] / df["J_catalog"]
    df["correction_factor"] = np.sqrt(df["J_catalog"] / df["J_raw"])
    df["rhos_corrected"] = df["correction_factor"] * df["rhos_raw"]
    df["rho_s_corrected"] = 4.0 * df["rhos_corrected"]
    df["theta_deg_from_list"] = np.degrees(
        np.arctan(
            df["rs_kpc"].to_numpy(dtype=float)
            / df["d_kpc"].to_numpy(dtype=float)
        )
    )

    return df


# ============================================================
# PLOTS
# ============================================================

def make_plots(df, paths):
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["theta_deg_from_list"], df["R_raw"], s=10)
    plt.xscale("log")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Angular size [deg]")
    plt.ylabel("R_raw = J_raw / J_catalog")
    plt.tight_layout()
    plt.savefig(paths["plot_R_vs_theta"], dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["theta_deg_from_list"], df["correction_factor"], s=10)
    plt.xscale("log")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Angular size [deg]")
    plt.ylabel("correction_factor = sqrt(J_catalog / J_raw)")
    plt.tight_layout()
    plt.savefig(paths["plot_X_vs_theta"], dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(df["R_raw"], bins=60)
    plt.axvline(1.0, linestyle="--")
    plt.xlabel("R_raw = J_raw / J_catalog")
    plt.ylabel("Number of halos")
    plt.tight_layout()
    plt.savefig(paths["plot_R_hist"], dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["d_kpc"], df["R_raw"], s=10)
    plt.xscale("log")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Distance [kpc]")
    plt.ylabel("R_raw = J_raw / J_catalog")
    plt.tight_layout()
    plt.savefig(paths["plot_R_vs_d"], dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["npix_touched"], df["R_raw"], s=10)
    plt.xscale("log")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("NPIX_TOUCHED")
    plt.ylabel("R_raw = J_raw / J_catalog")
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

    corr_map = corr_df.set_index("halo_name")["rhos_corrected"].to_dict()

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
        f"# Authoritative HDF5 catalogue: {paths['catalog']}",
        f"# Correction table: "
        f"{paths.get('metadata_output_table', paths['output_table'])}",
        "# Correction target: authoritative HDF5 catalogue Js",
        "# Correction rule:",
        "#   R_raw = J_raw / J_catalog",
        "#   correction_factor = sqrt(J_catalog / J_raw)",
        "#   rhos_corrected = rhos_raw * correction_factor",
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
    nside = args.nside

    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of two.")

    paths = get_paths(
        repop_id=repop_id,
        scenario=scenario,
        nside=nside,
        catalog_root=getattr(args, "catalog_root", DEFAULT_CATALOG_ROOT),
        clumpy_root=getattr(args, "clumpy_root", BASE_RUN_DIR),
    )
    raw_list_override = getattr(args, "raw_list", None)
    rendered_log_override = getattr(args, "raw_rendered_log", None)
    output_list_override = getattr(args, "output_list", None)
    output_dir_override = getattr(args, "output_dir", None)
    metadata_output_dir = getattr(args, "metadata_output_dir", None)
    if raw_list_override is not None:
        paths["input_list"] = raw_list_override
    if rendered_log_override is not None:
        paths["rendered_log"] = rendered_log_override
    if output_list_override is not None:
        paths["output_list"] = output_list_override
    if output_dir_override is not None:
        paths["output_dir"] = output_dir_override
        paths["output_table"] = (
            output_dir_override
            / f"matched_R_diagnostics_{scenario}_{paths['repop_tag']}"
            f"{nside_suffix(nside)}.csv"
        )
        paths["plot_R_vs_theta"] = output_dir_override / "R_vs_theta.png"
        paths["plot_X_vs_theta"] = output_dir_override / "X_vs_theta.png"
        paths["plot_R_hist"] = output_dir_override / "R_hist.png"
        paths["plot_R_vs_d"] = output_dir_override / "R_vs_distance.png"
        paths["plot_R_vs_npix"] = output_dir_override / "R_vs_npix.png"
    if metadata_output_dir is not None:
        paths["metadata_output_table"] = (
            metadata_output_dir
            / f"matched_R_diagnostics_{scenario}_{paths['repop_tag']}"
            f"{nside_suffix(nside)}.csv"
        )

    print()
    print("=" * 80)
    print("CLUMPY rho_s correction")
    print("=" * 80)
    print(f"Scenario: {scenario}")
    print(f"REPOP_ID: {repop_id}")
    print(f"REPOP_TAG: {paths['repop_tag']}")
    print(f"NSIDE: {nside}")
    print(f"Input list: {paths['input_list']}")
    print(f"Rendered log: {paths['rendered_log']}")
    print(f"Authoritative HDF5 catalogue: {paths['catalog']}")
    print(f"Output corrected list: {paths['output_list']}")
    print(f"Diagnostics output dir: {paths['output_dir']}")
    print("=" * 80)
    print()

    list_df = load_clumpy_list(paths["input_list"])
    rendered_df = load_rendered_log(paths["rendered_log"])
    validate_raw_correspondence(list_df, rendered_df)
    hdf5_rows, catalog_js = load_catalog_js_indexed(
        paths["catalog"],
        list_df["halo_name"].tolist(),
    )

    matched = build_matched_table(
        list_df,
        rendered_df,
        hdf5_rows,
        catalog_js,
    )
    matched.insert(0, "catalog_path", str(paths["catalog"]))
    matched.insert(
        1,
        "correction_target",
        "authoritative HDF5 catalogue Js",
    )

    if matched.empty:
        raise RuntimeError(
            "No matched valid halos found. Check halo names in list and rendered log."
        )

    if getattr(args, "validate_only", False):
        print(f"Validated matched raw halos: {len(matched)}")
        print("Validation-only mode: no correction products were written.")
        print("=" * 80)
        return

    paths["output_dir"].mkdir(parents=True, exist_ok=True)

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
    print(f"Median R_raw: {np.nanmedian(matched['R_raw']):.6f}")
    print(f"Mean R_raw:   {np.nanmean(matched['R_raw']):.6f}")
    print(
        "Median correction factor: "
        f"{np.nanmedian(matched['correction_factor']):.6f}"
    )
    print(
        "Mean correction factor:   "
        f"{np.nanmean(matched['correction_factor']):.6f}"
    )
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
