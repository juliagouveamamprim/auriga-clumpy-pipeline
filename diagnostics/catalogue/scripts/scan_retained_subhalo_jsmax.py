#!/usr/bin/env python3
"""Find the brightest retained subhalo in each completed CLUMPY case.

The diagnostic is deliberately product anchored:

* pointlike membership is replayed with the production validity mask and the
  effective ``THETACUT``/``JCPL`` values stored in the pointlike FITS;
* extended membership comes from the corrected external-halo list, after an
  exact ordered comparison with the corrected ``halo_rendered.log``.

Only catalogue metadata and chunk-sized slices are read.  The pointlike map
itself is never traversed and no production product is modified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
from astropy import units as u
from astropy.io import fits


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_subhalo_components import build_valid_mask  # noqa: E402


SCENARIOS = ("fragile", "resilient")
DEFAULT_NSIDE = 2048
DEFAULT_CHUNK_SIZE = 500_000
PROGRESS_INTERVAL_ROWS = 10_000_000
SCHEMA_VERSION = 1
REQUIRED_COLUMNS = (
    "Js",
    "D_Earth",
    "Vmax",
    "theta_s",
    "Cv",
    "r_s",
    "rho_s",
    "Xearth",
    "Yearth",
    "Zearth",
)
RETAINED_COLUMNS = (
    "halo_name",
    "hdf5_row",
    "component",
    "Js",
    "theta_s",
    "D_Earth",
    "Vmax",
    "r_s",
    "rho_s",
    "Cv",
    "Xearth",
    "Yearth",
    "Zearth",
)
OUTPUT_COLUMNS = (
    "schema_version",
    "scenario",
    "repop_id",
    "production_git_commit",
    "diagnostic_git_commit",
    "diagnostic_script_sha256",
    "nside",
    "theta_cut_deg",
    "pointlike_j_cut",
    "pointlike_cut_fraction",
    "j_pixel_ref",
    "extended_cut_fraction",
    "extended_j_cut",
    "pointlike_npoint_metadata",
    "retained_pointlike_count",
    "retained_extended_count",
    "retained_catalog_rows",
    "retained_catalog_sha256",
    "catalog_n_rows",
    "max_halo_name",
    "max_hdf5_row",
    "max_Js",
    "max_theta_s_deg",
    "max_D_Earth_kpc",
    "max_Vmax_kms",
    "max_r_s_kpc",
    "max_rho_s",
    "max_Cv",
    "max_Xearth_kpc",
    "max_Yearth_kpc",
    "max_Zearth_kpc",
    "max_classification",
    "Js_unit",
    "theta_s_unit",
    "D_Earth_unit",
    "Vmax_unit",
    "r_s_unit",
    "rho_s_unit",
    "Cv_unit",
    "Xearth_unit",
    "Yearth_unit",
    "Zearth_unit",
    "catalog_path",
    "retained_catalog_path",
    "pointlike_fits_path",
    "corrected_list_path",
    "corrected_rendered_log_path",
)


class DiagnosticError(RuntimeError):
    """A required production product is absent or inconsistent."""


@dataclass(frozen=True)
class CasePaths:
    catalog: Path
    pointlike_fits: Path
    corrected_list: Path
    corrected_rendered_log: Path
    retained_catalog: Path
    output_csv: Path


@dataclass(frozen=True)
class PointlikeMetadata:
    nside: int
    theta_cut_deg: float
    j_cut: float
    npoint: int
    cut_fraction: float
    j_pixel_ref: float
    extended_cut_fraction: float | None
    extended_j_cut: float | None


@dataclass(frozen=True)
class ListRow:
    name: str
    halo_type: str


@dataclass(frozen=True)
class Brightest:
    name: str
    row: int
    js: float
    theta_s_deg: float
    d_earth_kpc: float
    vmax_kms: float
    r_s_kpc: float
    rho_s: float
    cv: float
    xearth_kpc: float
    yearth_kpc: float
    zearth_kpc: float
    classification: str


def _decode_strings(values: Iterable[object]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise DiagnosticError(f"{label} is missing or not a regular file: {path}")
    if path.stat().st_size == 0:
        raise DiagnosticError(f"{label} is empty: {path}")


def case_paths(
    catalog_root: Path,
    clumpy_root: Path,
    output_dir: Path,
    scenario: str,
    repop_id: int,
    nside: int,
) -> CasePaths:
    tag = f"repop_{repop_id:04d}"
    run_tag = f"{tag}_nside{nside}"
    basename = f"annihil_gal2D_LOS0_0_FOV360x180_nside{nside}"
    scenario_root = Path(clumpy_root) / scenario
    return CasePaths(
        catalog=Path(catalog_root) / tag / f"fullrepop_hydro_{scenario}.h5",
        pointlike_fits=(
            scenario_root / "pointlike" / f"{tag}_pointlike_nside{nside}.fits"
        ),
        corrected_list=(
            scenario_root
            / "lists"
            / "corrected"
            / f"{tag}_rhocorr_nside{nside}.txt"
        ),
        corrected_rendered_log=(
            scenario_root
            / "outputs"
            / "corrected_clumpy"
            / run_tag
            / f"{basename}.halo_rendered.log"
        ),
        retained_catalog=(
            Path(output_dir)
            / f"retained_subhalos_{scenario}_{tag}_nside{nside}.csv.gz"
        ),
        output_csv=(
            Path(output_dir) / f"retained_jsmax_{scenario}_{tag}_nside{nside}.csv"
        ),
    )


def _finite_header_float(header: fits.Header, key: str, label: str) -> float:
    try:
        value = float(header[key])
    except (KeyError, TypeError, ValueError) as error:
        raise DiagnosticError(f"{label}: missing or invalid FITS keyword {key}") from error
    if not np.isfinite(value):
        raise DiagnosticError(f"{label}: FITS keyword {key} must be finite")
    return value


def load_pointlike_metadata(
    path: Path,
    scenario: str,
    repop_id: int,
    nside: int,
) -> PointlikeMetadata:
    """Read pointlike provenance without traversing either map column."""
    _require_nonempty_file(path, "pointlike FITS")
    try:
        with fits.open(path, mode="readonly", memmap=True, lazy_load_hdus=False) as hdul:
            if len(hdul) < 3:
                raise DiagnosticError(f"pointlike FITS has fewer than three HDUs: {path}")
            headers = []
            for index, expected_name in ((1, "JFACTOR"), (2, "JFACTOR_PER_SR")):
                hdu = hdul[index]
                label = f"{path} HDU {index}"
                if hdu.name != expected_name:
                    raise DiagnosticError(
                        f"{label}: expected EXTNAME={expected_name!r}, found {hdu.name!r}"
                    )
                header = hdu.header
                expected = {
                    "NSIDE": nside,
                    "SCENARIO": scenario,
                    "REPOPID": repop_id,
                }
                for key, value in expected.items():
                    if header.get(key) != value:
                        raise DiagnosticError(
                            f"{label}: expected {key}={value!r}, found {header.get(key)!r}"
                        )
                if header.get("CUTS") is not True:
                    raise DiagnosticError(f"{label}: expected CUTS=True production metadata")
                headers.append(header)

            first, second = headers
            required_keys = ("THETACUT", "JCPL", "NPOINT", "FPL", "JREF")
            for key in required_keys:
                if first.get(key) != second.get(key):
                    raise DiagnosticError(
                        f"{path}: pointlike metadata {key} differs between map HDUs"
                    )

            theta_cut = _finite_header_float(first, "THETACUT", str(path))
            j_cut = _finite_header_float(first, "JCPL", str(path))
            cut_fraction = _finite_header_float(first, "FPL", str(path))
            j_pixel_ref = _finite_header_float(first, "JREF", str(path))
            npoint = first.get("NPOINT")
            if not isinstance(npoint, (int, np.integer)) or npoint < 0:
                raise DiagnosticError(f"{path}: NPOINT must be a non-negative integer")
            if theta_cut <= 0.0 or j_cut < 0.0 or cut_fraction < 0.0 or j_pixel_ref <= 0.0:
                raise DiagnosticError(f"{path}: invalid non-positive pointlike cut metadata")
            if not np.isclose(j_cut, cut_fraction * j_pixel_ref, rtol=1e-12, atol=0.0):
                raise DiagnosticError(f"{path}: JCPL is inconsistent with FPL * JREF")

            fext = first.get("FEXT")
            jcext = first.get("JCEXT")
            if fext != second.get("FEXT") or jcext != second.get("JCEXT"):
                raise DiagnosticError(
                    f"{path}: extended cut metadata differs between map HDUs"
                )
            if (fext is None) != (jcext is None):
                raise DiagnosticError(f"{path}: FEXT and JCEXT must either both exist or both be absent")
            extended_fraction = None
            extended_cut = None
            if fext is not None:
                extended_fraction = _finite_header_float(first, "FEXT", str(path))
                extended_cut = _finite_header_float(first, "JCEXT", str(path))

    except OSError as error:
        raise DiagnosticError(f"could not open pointlike FITS {path}: {error}") from error

    return PointlikeMetadata(
        nside=nside,
        theta_cut_deg=theta_cut,
        j_cut=j_cut,
        npoint=int(npoint),
        cut_fraction=cut_fraction,
        j_pixel_ref=j_pixel_ref,
        extended_cut_fraction=extended_fraction,
        extended_j_cut=extended_cut,
    )


def load_corrected_list(
    path: Path,
    scenario: str,
    repop_id: int,
    nside: int,
    catalog_path: Path,
) -> list[ListRow]:
    _require_nonempty_file(path, "corrected external-halo list")
    rows: list[ListRow] = []
    metadata: dict[str, str] = {}
    prefixes = {
        "Scenario:": "scenario",
        "REPOP_ID:": "repop_id",
        "HEALPix NSIDE:": "nside",
        "HDF5 internal iteration:": "iteration",
        "Source HDF5:": "source_hdf5",
    }
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_text = stripped[1:].strip()
                for prefix, key in prefixes.items():
                    if not header_text.startswith(prefix):
                        continue
                    value = header_text[len(prefix):].strip()
                    if key in metadata and metadata[key] != value:
                        raise DiagnosticError(
                            f"conflicting corrected-list {key} metadata in {path}"
                        )
                    metadata[key] = value
                    break
                continue
            fields = stripped.split()
            if len(fields) != 13:
                raise DiagnosticError(
                    f"malformed corrected-list row {path}:{line_number}: "
                    f"expected 13 fields, found {len(fields)}"
                )
            rows.append(ListRow(fields[0], fields[1]))
    if not rows:
        raise DiagnosticError(f"corrected external-halo list has no rows: {path}")
    names = [row.name for row in rows]
    if len(names) != len(set(names)):
        raise DiagnosticError(f"corrected external-halo list contains duplicate names: {path}")

    required = {
        "scenario": scenario,
        "repop_id": str(repop_id),
        "nside": str(nside),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise DiagnosticError(
                f"corrected-list {key} metadata mismatch in {path}: "
                f"expected {expected!r}, found {metadata.get(key)!r}"
            )
    if "iteration" in metadata and metadata["iteration"] != "0":
        raise DiagnosticError(
            f"corrected-list HDF5 iteration mismatch in {path}: "
            f"expected '0', found {metadata['iteration']!r}"
        )
    if "source_hdf5" in metadata:
        source_path = Path(metadata["source_hdf5"])
        if source_path.name != Path(catalog_path).name:
            raise DiagnosticError(
                f"corrected-list source catalogue mismatch in {path}: "
                f"expected filename {Path(catalog_path).name!r}, "
                f"found {source_path.name!r}"
            )
        expected_parent = f"repop_{repop_id:04d}"
        if source_path.parent != Path(".") and source_path.parent.name.startswith("repop_"):
            if source_path.parent.name != expected_parent:
                raise DiagnosticError(
                    f"corrected-list source repopulation mismatch in {path}: "
                    f"expected {expected_parent!r}, found {source_path.parent.name!r}"
                )
    return rows


def validate_corrected_rendered_log(path: Path, list_rows: Sequence[ListRow]) -> None:
    _require_nonempty_file(path, "corrected halo_rendered.log")
    rendered: list[tuple[int, str, str]] = []
    header_seen = False
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if tuple(stripped[1:].strip().split()) == (
                    "HALO_INDEX",
                    "NAME",
                    "TYPE",
                    "J_RENDERED",
                    "NPIX_TOUCHED",
                ):
                    header_seen = True
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise DiagnosticError(
                    f"malformed rendered-log row {path}:{line_number}: "
                    f"expected 5 fields, found {len(fields)}"
                )
            try:
                index = int(fields[0])
                j_rendered = float(fields[3])
                npix = int(fields[4])
            except ValueError as error:
                raise DiagnosticError(
                    f"malformed rendered-log value {path}:{line_number}"
                ) from error
            if not np.isfinite(j_rendered) or j_rendered < 0.0 or npix < 0:
                raise DiagnosticError(f"invalid rendered-log values {path}:{line_number}")
            rendered.append((index, fields[1], fields[2]))

    if not header_seen:
        raise DiagnosticError(f"corrected rendered log is missing its expected header: {path}")
    if len(rendered) != len(list_rows):
        raise DiagnosticError(
            f"corrected list/log count mismatch: list={len(list_rows)}, log={len(rendered)}"
        )
    for expected_index, (list_row, log_row) in enumerate(zip(list_rows, rendered)):
        index, name, halo_type = log_row
        if index != expected_index:
            raise DiagnosticError(
                f"corrected rendered-log index mismatch at row {expected_index}: {index}"
            )
        if name != list_row.name or halo_type != list_row.halo_type:
            raise DiagnosticError(
                f"corrected list/log identity mismatch at row {expected_index}: "
                f"list=({list_row.name}, {list_row.halo_type}), "
                f"log=({name}, {halo_type})"
            )


def _validate_unit(unit_text: str, expected: u.UnitBase, label: str) -> None:
    try:
        unit = u.Unit(unit_text)
    except (TypeError, ValueError) as error:
        raise DiagnosticError(f"invalid {label} unit {unit_text!r}") from error
    if not unit.is_equivalent(expected):
        raise DiagnosticError(
            f"unexpected {label} unit {unit_text!r}; expected compatibility with {expected}"
        )


def _prefer_candidate(current: Brightest | None, candidate: Brightest) -> Brightest:
    if current is None:
        return candidate
    if candidate.js > current.js or (
        candidate.js == current.js and candidate.row < current.row
    ):
        return candidate
    return current


def scan_catalog(
    catalog_path: Path,
    extended_rows: Sequence[ListRow],
    metadata: PointlikeMetadata,
    chunk_size: int,
    retained_writer: csv.DictWriter,
) -> tuple[Brightest, int, int, dict[str, str]]:
    """Replay selection, stream retained rows, and find the brightest in one pass."""
    _require_nonempty_file(catalog_path, "HDF5 catalogue")
    extended_names = {row.name for row in extended_rows}
    found_extended: set[str] = set()
    retained_pointlike = 0
    brightest: Brightest | None = None

    try:
        with h5py.File(catalog_path, "r") as handle:
            if "iteration_0" not in handle:
                raise DiagnosticError(f"missing iteration_0 in {catalog_path}")
            group = handle["iteration_0"]
            if "data" not in group or "halo_name" not in group:
                raise DiagnosticError(
                    f"missing iteration_0/data or iteration_0/halo_name in {catalog_path}"
                )
            data = group["data"]
            stored_names = group["halo_name"]
            if data.ndim != 2 or stored_names.ndim != 1:
                raise DiagnosticError(f"invalid catalogue dataset dimensions in {catalog_path}")
            if data.shape[0] != stored_names.shape[0]:
                raise DiagnosticError(f"catalogue data/name length mismatch in {catalog_path}")
            n_saved = group.attrs.get("n_saved")
            if n_saved is None or int(n_saved) != data.shape[0]:
                raise DiagnosticError(f"catalogue n_saved metadata mismatch in {catalog_path}")
            if "column_names" not in data.attrs or "units" not in data.attrs:
                raise DiagnosticError(f"catalogue is missing column_names or units in {catalog_path}")
            column_names = _decode_strings(data.attrs["column_names"])
            units = _decode_strings(data.attrs["units"])
            if len(column_names) != data.shape[1] or len(units) != data.shape[1]:
                raise DiagnosticError(f"catalogue column metadata length mismatch in {catalog_path}")
            missing = [name for name in REQUIRED_COLUMNS if column_names.count(name) != 1]
            if missing:
                raise DiagnosticError(
                    f"catalogue must contain each required column exactly once; invalid: {missing}"
                )
            indices = {name: column_names.index(name) for name in REQUIRED_COLUMNS}
            unit_map = {name: units[index] for name, index in indices.items()}
            _validate_unit(unit_map["Js"], u.GeV**2 / u.cm**5, "Js")
            _validate_unit(unit_map["theta_s"], u.deg, "theta_s")
            _validate_unit(unit_map["D_Earth"], u.kpc, "D_Earth")
            _validate_unit(unit_map["Vmax"], u.km / u.s, "Vmax")
            _validate_unit(unit_map["r_s"], u.kpc, "r_s")
            _validate_unit(unit_map["rho_s"], u.Msun / u.kpc**3, "rho_s")
            _validate_unit(unit_map["Cv"], u.dimensionless_unscaled, "Cv")
            for coordinate in ("Xearth", "Yearth", "Zearth"):
                _validate_unit(unit_map[coordinate], u.kpc, coordinate)

            next_progress_row = PROGRESS_INTERVAL_ROWS
            for start in range(0, data.shape[0], chunk_size):
                stop = min(start + chunk_size, data.shape[0])
                arr = np.asarray(data[start:stop])
                names = _decode_strings(stored_names[start:stop])
                for local_index, name in enumerate(names):
                    row_index = start + local_index
                    expected_name = f"hydro_res_{row_index + 1:05d}"
                    if name != expected_name:
                        raise DiagnosticError(
                            f"HDF5 identity mismatch at row {row_index} in "
                            f"{catalog_path}: expected {expected_name!r}, "
                            f"found {name!r}"
                        )
                js = np.asarray(arr[:, indices["Js"]], dtype=np.float64)
                d_earth = np.asarray(arr[:, indices["D_Earth"]], dtype=np.float64)
                vmax = np.asarray(arr[:, indices["Vmax"]], dtype=np.float64)
                theta_s = np.asarray(arr[:, indices["theta_s"]], dtype=np.float64)
                cv = np.asarray(arr[:, indices["Cv"]], dtype=np.float64)
                r_s = np.asarray(arr[:, indices["r_s"]], dtype=np.float64)
                rho_s = np.asarray(arr[:, indices["rho_s"]], dtype=np.float64)
                x_e = np.asarray(arr[:, indices["Xearth"]], dtype=np.float64)
                y_e = np.asarray(arr[:, indices["Yearth"]], dtype=np.float64)
                z_e = np.asarray(arr[:, indices["Zearth"]], dtype=np.float64)

                valid = build_valid_mask(
                    js=js,
                    d_earth=d_earth,
                    theta_s=theta_s,
                    r_s=r_s,
                    rho_s=rho_s,
                    x_e=x_e,
                    y_e=y_e,
                    z_e=z_e,
                )
                pointlike = (
                    valid
                    & (theta_s < metadata.theta_cut_deg)
                    & (js >= metadata.j_cut)
                )
                retained_pointlike += int(np.count_nonzero(pointlike))

                extended = np.fromiter(
                    (name in extended_names for name in names),
                    dtype=bool,
                    count=len(names),
                )
                overlap = pointlike & extended
                if np.any(overlap):
                    name = names[int(np.flatnonzero(overlap)[0])]
                    raise DiagnosticError(
                        f"halo {name!r} is selected as both pointlike and extended"
                    )

                for local_index in np.flatnonzero(pointlike | extended):
                    local_index = int(local_index)
                    name = names[local_index]
                    component = "extended" if extended[local_index] else "pointlike"
                    if component == "extended":
                        if name in found_extended:
                            raise DiagnosticError(
                                f"extended halo {name!r} occurs more than once in {catalog_path}"
                            )
                        found_extended.add(name)
                        values = (
                            js[local_index],
                            theta_s[local_index],
                            d_earth[local_index],
                        )
                        if not all(np.isfinite(value) for value in values):
                            raise DiagnosticError(
                                f"extended halo {name!r} has non-finite properties"
                            )
                        if js[local_index] <= 0.0 or d_earth[local_index] <= 0.0:
                            raise DiagnosticError(
                                f"extended halo {name!r} has invalid properties"
                            )

                    row_index = start + local_index
                    retained_row = {
                        "halo_name": name,
                        "hdf5_row": row_index,
                        "component": component,
                        "Js": float(js[local_index]),
                        "theta_s": float(theta_s[local_index]),
                        "D_Earth": float(d_earth[local_index]),
                        "Vmax": float(vmax[local_index]),
                        "r_s": float(r_s[local_index]),
                        "rho_s": float(rho_s[local_index]),
                        "Cv": float(cv[local_index]),
                        "Xearth": float(x_e[local_index]),
                        "Yearth": float(y_e[local_index]),
                        "Zearth": float(z_e[local_index]),
                    }
                    retained_writer.writerow(retained_row)
                    brightest = _prefer_candidate(
                        brightest,
                        Brightest(
                            name=name,
                            row=row_index,
                            js=float(js[local_index]),
                            theta_s_deg=float(theta_s[local_index]),
                            d_earth_kpc=float(d_earth[local_index]),
                            vmax_kms=float(vmax[local_index]),
                            r_s_kpc=float(r_s[local_index]),
                            rho_s=float(rho_s[local_index]),
                            cv=float(cv[local_index]),
                            xearth_kpc=float(x_e[local_index]),
                            yearth_kpc=float(y_e[local_index]),
                            zearth_kpc=float(z_e[local_index]),
                            classification=component,
                        ),
                    )

                if stop >= next_progress_row:
                    percentage = 100.0 * stop / data.shape[0]
                    print(
                        f"progress: processed={stop:,} rows, "
                        f"total={data.shape[0]:,} rows "
                        f"({percentage:.1f}%)",
                        flush=True,
                    )
                    next_progress_row = (
                        stop // PROGRESS_INTERVAL_ROWS + 1
                    ) * PROGRESS_INTERVAL_ROWS

            catalog_n_rows = int(data.shape[0])
    except OSError as error:
        raise DiagnosticError(f"could not read HDF5 catalogue {catalog_path}: {error}") from error

    missing_extended = extended_names - found_extended
    if missing_extended:
        examples = ", ".join(sorted(missing_extended)[:5])
        raise DiagnosticError(
            f"{len(missing_extended)} corrected-list halo(s) are absent from the catalogue; "
            f"examples: {examples}"
        )
    if retained_pointlike != metadata.npoint:
        raise DiagnosticError(
            f"retained pointlike count mismatch: selected={retained_pointlike}, "
            f"FITS NPOINT={metadata.npoint}"
        )
    if brightest is None:
        raise DiagnosticError("no retained pointlike or extended subhalo was found")

    return brightest, retained_pointlike, catalog_n_rows, unit_map


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _script_sha256() -> str:
    return _file_sha256(Path(__file__))


def build_output_row(
    paths: CasePaths,
    scenario: str,
    repop_id: int,
    production_git_commit: str,
    diagnostic_git_commit: str,
    metadata: PointlikeMetadata,
    brightest: Brightest,
    retained_pointlike: int,
    retained_extended: int,
    retained_catalog_sha256: str,
    catalog_n_rows: int,
    units: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "repop_id": repop_id,
        "production_git_commit": production_git_commit,
        "diagnostic_git_commit": diagnostic_git_commit,
        "diagnostic_script_sha256": _script_sha256(),
        "nside": metadata.nside,
        "theta_cut_deg": format(metadata.theta_cut_deg, ".17g"),
        "pointlike_j_cut": format(metadata.j_cut, ".17g"),
        "pointlike_cut_fraction": format(metadata.cut_fraction, ".17g"),
        "j_pixel_ref": format(metadata.j_pixel_ref, ".17g"),
        "extended_cut_fraction": "" if metadata.extended_cut_fraction is None else format(metadata.extended_cut_fraction, ".17g"),
        "extended_j_cut": "" if metadata.extended_j_cut is None else format(metadata.extended_j_cut, ".17g"),
        "pointlike_npoint_metadata": metadata.npoint,
        "retained_pointlike_count": retained_pointlike,
        "retained_extended_count": retained_extended,
        "retained_catalog_rows": retained_pointlike + retained_extended,
        "retained_catalog_sha256": retained_catalog_sha256,
        "catalog_n_rows": catalog_n_rows,
        "max_halo_name": brightest.name,
        "max_hdf5_row": brightest.row,
        "max_Js": format(brightest.js, ".17g"),
        "max_theta_s_deg": format(brightest.theta_s_deg, ".17g"),
        "max_D_Earth_kpc": format(brightest.d_earth_kpc, ".17g"),
        "max_Vmax_kms": format(brightest.vmax_kms, ".17g"),
        "max_r_s_kpc": format(brightest.r_s_kpc, ".17g"),
        "max_rho_s": format(brightest.rho_s, ".17g"),
        "max_Cv": format(brightest.cv, ".17g"),
        "max_Xearth_kpc": format(brightest.xearth_kpc, ".17g"),
        "max_Yearth_kpc": format(brightest.yearth_kpc, ".17g"),
        "max_Zearth_kpc": format(brightest.zearth_kpc, ".17g"),
        "max_classification": brightest.classification,
        "Js_unit": units["Js"],
        "theta_s_unit": units["theta_s"],
        "D_Earth_unit": units["D_Earth"],
        "Vmax_unit": units["Vmax"],
        "r_s_unit": units["r_s"],
        "rho_s_unit": units["rho_s"],
        "Cv_unit": units["Cv"],
        "Xearth_unit": units["Xearth"],
        "Yearth_unit": units["Yearth"],
        "Zearth_unit": units["Zearth"],
        "catalog_path": str(paths.catalog.resolve()),
        "retained_catalog_path": str(paths.retained_catalog.resolve()),
        "pointlike_fits_path": str(paths.pointlike_fits.resolve()),
        "corrected_list_path": str(paths.corrected_list.resolve()),
        "corrected_rendered_log_path": str(paths.corrected_rendered_log.resolve()),
    }


def _atomic_write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def validate_existing_output(
    path: Path,
    paths: CasePaths,
    scenario: str,
    repop_id: int,
    nside: int,
    production_git_commit: str,
) -> dict[str, str]:
    _require_nonempty_file(path, "existing case CSV")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise DiagnosticError(f"existing case CSV has an unexpected schema: {path}")
        rows = list(reader)
    if len(rows) != 1:
        raise DiagnosticError(f"existing case CSV must contain exactly one row: {path}")
    row = rows[0]
    expected = {
        "schema_version": str(SCHEMA_VERSION),
        "scenario": scenario,
        "repop_id": str(repop_id),
        "nside": str(nside),
        "production_git_commit": production_git_commit,
        "catalog_path": str(paths.catalog.resolve()),
        "retained_catalog_path": str(paths.retained_catalog.resolve()),
        "pointlike_fits_path": str(paths.pointlike_fits.resolve()),
        "corrected_list_path": str(paths.corrected_list.resolve()),
        "corrected_rendered_log_path": str(paths.corrected_rendered_log.resolve()),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise DiagnosticError(
                f"existing case CSV {key} mismatch in {path}: "
                f"expected {value!r}, found {row.get(key)!r}"
            )
    return row


def validate_retained_catalog(path: Path, summary: dict[str, str]) -> None:
    """Validate a completed retained catalogue against its summary checkpoint."""
    _require_nonempty_file(path, "existing retained catalogue")
    expected_digest = summary.get("retained_catalog_sha256", "")
    actual_digest = _file_sha256(path)
    if actual_digest != expected_digest:
        raise DiagnosticError(
            f"retained catalogue checksum mismatch for {path}: "
            f"expected {expected_digest!r}, found {actual_digest!r}"
        )

    counts = {"pointlike": 0, "extended": 0}
    brightest: dict[str, str] | None = None
    previous_row = -1
    try:
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != RETAINED_COLUMNS:
                raise DiagnosticError(
                    f"existing retained catalogue has an unexpected schema: {path}"
                )
            for row in reader:
                try:
                    hdf5_row = int(row["hdf5_row"])
                    js = float(row["Js"])
                except (KeyError, TypeError, ValueError) as error:
                    raise DiagnosticError(
                        f"invalid retained-catalogue row in {path}"
                    ) from error
                component = row.get("component")
                if component not in counts:
                    raise DiagnosticError(
                        f"invalid retained component {component!r} in {path}"
                    )
                if hdf5_row <= previous_row:
                    raise DiagnosticError(
                        f"retained catalogue rows are not strictly increasing in {path}"
                    )
                if not np.isfinite(js) or js <= 0.0:
                    raise DiagnosticError(f"invalid retained Js in {path}")
                previous_row = hdf5_row
                counts[component] += 1
                if brightest is None:
                    brightest = row
                else:
                    best_js = float(brightest["Js"])
                    best_row = int(brightest["hdf5_row"])
                    if js > best_js or (js == best_js and hdf5_row < best_row):
                        brightest = row
    except (OSError, EOFError) as error:
        raise DiagnosticError(f"could not read retained catalogue {path}: {error}") from error

    total = counts["pointlike"] + counts["extended"]
    expected_counts = {
        "retained_pointlike_count": counts["pointlike"],
        "retained_extended_count": counts["extended"],
        "retained_catalog_rows": total,
    }
    for key, actual in expected_counts.items():
        try:
            expected = int(summary[key])
        except (KeyError, ValueError) as error:
            raise DiagnosticError(f"invalid {key} in summary for {path}") from error
        if actual != expected:
            raise DiagnosticError(
                f"{key} mismatch for {path}: summary={expected}, catalogue={actual}"
            )
    if brightest is None:
        raise DiagnosticError(f"retained catalogue contains no rows: {path}")

    exact = {
        "max_halo_name": brightest["halo_name"],
        "max_hdf5_row": brightest["hdf5_row"],
        "max_classification": brightest["component"],
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise DiagnosticError(
                f"brightest-summary mismatch for {key} in {path}: "
                f"summary={summary.get(key)!r}, catalogue={expected!r}"
            )
    float_pairs = {
        "max_Js": "Js",
        "max_theta_s_deg": "theta_s",
        "max_D_Earth_kpc": "D_Earth",
        "max_Vmax_kms": "Vmax",
        "max_r_s_kpc": "r_s",
        "max_rho_s": "rho_s",
        "max_Cv": "Cv",
        "max_Xearth_kpc": "Xearth",
        "max_Yearth_kpc": "Yearth",
        "max_Zearth_kpc": "Zearth",
    }
    for summary_key, retained_key in float_pairs.items():
        if float(summary[summary_key]) != float(brightest[retained_key]):
            raise DiagnosticError(
                f"brightest-summary mismatch for {summary_key} in {path}"
            )


def process_case(
    paths: CasePaths,
    scenario: str,
    repop_id: int,
    nside: int,
    chunk_size: int,
    production_git_commit: str,
    diagnostic_git_commit: str,
    overwrite: bool = False,
) -> str:
    print(f"starting: {scenario} repop_{repop_id:04d}", flush=True)
    summary_exists = paths.output_csv.exists()
    retained_exists = paths.retained_catalog.exists()
    if summary_exists and retained_exists and not overwrite:
        summary = validate_existing_output(
            paths.output_csv,
            paths,
            scenario,
            repop_id,
            nside,
            production_git_commit,
        )
        validate_retained_catalog(paths.retained_catalog, summary)
        return "skipped"

    metadata = load_pointlike_metadata(
        paths.pointlike_fits, scenario, repop_id, nside
    )
    extended_rows = load_corrected_list(
        paths.corrected_list,
        scenario=scenario,
        repop_id=repop_id,
        nside=nside,
        catalog_path=paths.catalog,
    )
    validate_corrected_rendered_log(paths.corrected_rendered_log, extended_rows)
    paths.retained_catalog.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=paths.retained_catalog.parent,
        prefix=f".{paths.retained_catalog.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_catalog = Path(temporary_name)
    try:
        with temporary_catalog.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_stream,
                mtime=0,
            ) as compressed_stream:
                with io.TextIOWrapper(
                    compressed_stream,
                    encoding="utf-8",
                    newline="",
                ) as text_stream:
                    writer = csv.DictWriter(
                        text_stream,
                        fieldnames=RETAINED_COLUMNS,
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    brightest, retained_pointlike, catalog_n_rows, units = scan_catalog(
                        paths.catalog,
                        extended_rows,
                        metadata,
                        chunk_size,
                        writer,
                    )
            raw_stream.flush()
            os.fsync(raw_stream.fileno())

        retained_digest = _file_sha256(temporary_catalog)
        row = build_output_row(
            paths=paths,
            scenario=scenario,
            repop_id=repop_id,
            production_git_commit=production_git_commit,
            diagnostic_git_commit=diagnostic_git_commit,
            metadata=metadata,
            brightest=brightest,
            retained_pointlike=retained_pointlike,
            retained_extended=len(extended_rows),
            retained_catalog_sha256=retained_digest,
            catalog_n_rows=catalog_n_rows,
            units=units,
        )
        os.replace(temporary_catalog, paths.retained_catalog)
        _atomic_write_rows(paths.output_csv, [row])
    finally:
        if temporary_catalog.exists():
            temporary_catalog.unlink()
    print(
        f"completed: {scenario} repop_{repop_id:04d}: "
        f"retained pointlike={retained_pointlike:,}, extended={len(extended_rows):,}",
        flush=True,
    )
    return "written"


def merge_case_outputs(paths: Sequence[Path], output: Path) -> None:
    rows: list[dict[str, str]] = []
    for path in paths:
        _require_nonempty_file(path, "case CSV selected for merge")
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
                raise DiagnosticError(f"case CSV has an unexpected schema: {path}")
            current = list(reader)
        if len(current) != 1:
            raise DiagnosticError(f"case CSV must contain exactly one row: {path}")
        rows.append(current[0])
    scenario_order = {name: index for index, name in enumerate(SCENARIOS)}
    rows.sort(key=lambda row: (scenario_order[row["scenario"]], int(row["repop_id"])))
    identities = [(row["scenario"], row["repop_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise DiagnosticError("duplicate scenario/repopulation rows selected for merge")
    _atomic_write_rows(output, rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=(*SCENARIOS, "both"))
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=0, help="Inclusive final ID")
    parser.add_argument("--nside", type=int, default=DEFAULT_NSIDE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--catalog-root", type=Path, default=REPOSITORY_ROOT / "outputs")
    parser.add_argument(
        "--clumpy-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "clumpy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "diagnostics" / "catalogue" / "retained_jsmax",
    )
    parser.add_argument(
        "--production-git-commit",
        required=True,
        help="Audited production-pipeline commit recorded in every output row.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--merge-output",
        type=Path,
        default=None,
        help="Optionally merge the selected per-case CSVs after processing.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.start_id < 0 or args.end_id < 0 or args.start_id > args.end_id:
        raise ValueError("Require non-negative --start-id <= --end-id")
    if args.nside <= 0 or (args.nside & (args.nside - 1)) != 0:
        raise ValueError("--nside must be a positive power of two")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not args.production_git_commit.strip():
        raise ValueError("--production-git-commit must not be empty")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    scenarios = SCENARIOS if args.scenario == "both" else (args.scenario,)
    diagnostic_commit = _git_commit()
    selected_outputs: list[Path] = []

    for scenario in scenarios:
        for repop_id in range(args.start_id, args.end_id + 1):
            paths = case_paths(
                args.catalog_root,
                args.clumpy_root,
                args.output_dir,
                scenario,
                repop_id,
                args.nside,
            )
            status = process_case(
                paths=paths,
                scenario=scenario,
                repop_id=repop_id,
                nside=args.nside,
                chunk_size=args.chunk_size,
                production_git_commit=args.production_git_commit,
                diagnostic_git_commit=diagnostic_commit,
                overwrite=args.overwrite,
            )
            selected_outputs.append(paths.output_csv)
            print(f"{status}: {scenario} repop_{repop_id:04d}: {paths.output_csv}")

    if args.merge_output is not None:
        merge_case_outputs(selected_outputs, args.merge_output)
        print(f"merged: {args.merge_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
