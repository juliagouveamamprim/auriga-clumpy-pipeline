#!/usr/bin/env python3
"""Combine validated dark-only catalogue-cut results for Figure 9."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np


SCENARIOS = ("fragile", "resilient")
EXPECTED_REPOP_IDS = frozenset(range(50))
EXPECTED_THRESHOLDS = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
EXISTING_THRESHOLDS = (1.0e-3,)
ADDITIONAL_THRESHOLDS = (1.0e-5, 1.0e-4, 1.0e-2)
EXPECTED_POPULATION_SELECTION = "dark_vmax_le_8kms"
EXPECTED_MAX_VMAX_KM_S = 8.0
EXPECTED_NSIDE = 2048

# Kept in step with plot_multi_repop_cut_diagnostics.py.  Requiring these
# before writing makes the output directly consumable by that plotter.
PLOTTING_COLUMNS = {
    "scenario",
    "repop_id",
    "nside",
    "pointlike_f",
    "extended_f",
    "j_pixel_ref",
    "fraction_discarded_js_to_full",
    "n_pointlike_total",
    "n_pointlike_kept",
    "n_pointlike_discarded",
    "n_extended_total",
    "n_extended_kept",
    "n_extended_discarded",
}
REFERENCE_RATIO_COLUMN = (
    "ratio_max_discarded_theta_s_envelope_to_j_pixel_ref"
)
DISCARDED_PEAK_COLUMN = (
    "max_discarded_combined_theta_s_envelope_pixel"
)

# These are emitted by the dark-only implementation at commit 4c65a21.
DARK_METADATA_COLUMNS = {
    "population_selection",
    "max_vmax_km_s",
    "npix",
    "theta_pix_deg",
    "theta_min_deg",
    "theta_aperture_deg",
    "extended_envelope_modes",
}
JS_TOTAL_COLUMNS = {
    "pointlike_full_js_sum",
    "extended_full_js_sum",
    "full_js_sum",
}
INTEGER_INVARIANT_COLUMNS = (
    "n_physical_valid_total",
    "n_vmax_excluded",
    "n_valid_total",
    "n_pointlike_total",
    "n_extended_total",
)
FLOAT_INVARIANT_COLUMNS = (
    "j_pixel_ref",
    "brightest_pointlike_proxy",
    "brightest_extended_proxy",
    "pointlike_full_js_sum",
    "extended_full_js_sum",
    "full_js_sum",
    "full_combined_max",
)
STRING_INVARIANT_COLUMNS = ("input_h5",)
GLOBAL_INTEGER_CONFIG_COLUMNS = ("nside", "npix")
GLOBAL_FLOAT_CONFIG_COLUMNS = (
    "theta_pix_deg",
    "theta_min_deg",
    "theta_aperture_deg",
)
GLOBAL_STRING_CONFIG_COLUMNS = ("extended_envelope_modes",)

THRESHOLD_RTOL = 1.0e-12
FLOAT_RTOL = 1.0e-10
FLOAT_ATOL = 0.0
CONFIG_RTOL = 1.0e-12
CONFIG_ATOL = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the archived f=1e-3 and additional-threshold "
            "dark-only Figure 9 validation CSVs."
        )
    )
    parser.add_argument("--fragile-existing-csv", type=Path, required=True)
    parser.add_argument("--fragile-additional-csv", type=Path, required=True)
    parser.add_argument("--resilient-existing-csv", type=Path, required=True)
    parser.add_argument("--resilient-additional-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output CSV.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Input CSV has duplicate columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return list(reader.fieldnames), rows


def _float(row: dict[str, str], column: str, context: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {column!r} in {context}") from error
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {column!r} in {context}: {value!r}")
    return value


def _integer(row: dict[str, str], column: str, context: str) -> int:
    value = _float(row, column, context)
    integer = int(value)
    if value != integer:
        raise ValueError(f"Non-integer {column!r} in {context}: {value!r}")
    return integer


def _canonical_threshold(value: float, context: str) -> float:
    matches = [
        expected
        for expected in EXPECTED_THRESHOLDS
        if np.isclose(
            value, expected, rtol=THRESHOLD_RTOL, atol=0.0
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Unexpected symmetric threshold {value!r} in {context}; "
            f"expected one of {EXPECTED_THRESHOLDS}"
        )
    return matches[0]


def _validate_input_rows(
    path: Path,
    rows: list[dict[str, str]],
    scenario: str,
) -> None:
    required = PLOTTING_COLUMNS | DARK_METADATA_COLUMNS | JS_TOTAL_COLUMNS
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(
            f"Input CSV {path} is missing required columns: "
            + ", ".join(missing)
        )
    if (
        REFERENCE_RATIO_COLUMN not in rows[0]
        and DISCARDED_PEAK_COLUMN not in rows[0]
    ):
        raise ValueError(
            f"Input CSV {path} lacks the conservative accumulated-impact "
            "metric required by the Figure 9 plotter"
        )

    for line_number, row in enumerate(rows, start=2):
        context = f"{path}, row {line_number}"
        if row["scenario"] != scenario:
            raise ValueError(
                f"Unexpected scenario in {context}: {row['scenario']!r}; "
                f"expected {scenario!r}"
            )
        if row["population_selection"] != EXPECTED_POPULATION_SELECTION:
            raise ValueError(
                f"Inconsistent population_selection in {context}: "
                f"{row['population_selection']!r}; expected "
                f"{EXPECTED_POPULATION_SELECTION!r}"
            )
        vmax = _float(row, "max_vmax_km_s", context)
        if not np.isclose(
            vmax, EXPECTED_MAX_VMAX_KM_S, rtol=0.0, atol=0.0
        ):
            raise ValueError(
                f"Inconsistent max_vmax_km_s in {context}: {vmax!r}; "
                f"expected {EXPECTED_MAX_VMAX_KM_S:g}"
            )
        nside = _integer(row, "nside", context)
        if nside != EXPECTED_NSIDE:
            raise ValueError(
                f"Inconsistent NSIDE in {context}: {nside}; "
                f"expected {EXPECTED_NSIDE}"
            )
        repop_id = _integer(row, "repop_id", context)
        if repop_id < 0:
            raise ValueError(f"Negative repop_id in {context}: {repop_id}")
        for column in JS_TOTAL_COLUMNS | {"j_pixel_ref"}:
            value = _float(row, column, context)
            if value <= 0.0:
                raise ValueError(
                    f"{column!r} must be positive in {context}: {value!r}"
                )
        for column in (
            "n_pointlike_total",
            "n_pointlike_kept",
            "n_pointlike_discarded",
            "n_extended_total",
            "n_extended_kept",
            "n_extended_discarded",
        ):
            if _integer(row, column, context) < 0:
                raise ValueError(
                    f"{column!r} must be non-negative in {context}"
                )
        _float(row, "fraction_discarded_js_to_full", context)


def _diagonal_rows(
    rows: Iterable[dict[str, str]], path: Path
) -> list[tuple[dict[str, str], int, float]]:
    selected = []
    for line_number, row in enumerate(rows, start=2):
        context = f"{path}, row {line_number}"
        pointlike_f = _float(row, "pointlike_f", context)
        extended_f = _float(row, "extended_f", context)
        if not np.isclose(
            pointlike_f,
            extended_f,
            rtol=THRESHOLD_RTOL,
            atol=0.0,
        ):
            continue
        threshold = _canonical_threshold(pointlike_f, context)
        selected.append(
            (row, _integer(row, "repop_id", context), threshold)
        )
    if not selected:
        raise ValueError(f"No symmetric cut rows found in {path}")
    return selected


def _validate_expected_source_thresholds(
    selected: list[tuple[dict[str, str], int, float]],
    expected: tuple[float, ...],
    path: Path,
) -> None:
    observed = {threshold for _, _, threshold in selected}
    if observed != set(expected):
        raise ValueError(
            f"Wrong symmetric thresholds in {path}: found "
            f"{sorted(observed)}, expected {sorted(expected)}"
        )

    repop_ids = {repop_id for _, repop_id, _ in selected}
    if repop_ids != EXPECTED_REPOP_IDS:
        missing = sorted(EXPECTED_REPOP_IDS - repop_ids)
        unexpected = sorted(repop_ids - EXPECTED_REPOP_IDS)
        raise ValueError(
            f"Wrong repopulations in {path}: missing={missing}, "
            f"unexpected={unexpected}; expected 0000-0049"
        )


def _validate_constant_column(
    rows: list[dict[str, str]],
    column: str,
    *,
    numeric: str | None = None,
) -> None:
    present = [row.get(column, "") != "" for row in rows]
    if any(present) and not all(present):
        raise ValueError(
            f"Configuration column {column!r} is missing from only some rows"
        )
    if not any(present):
        return
    context = f"combined selected rows ({column})"
    if numeric == "integer":
        values = [_integer(row, column, context) for row in rows]
        consistent = len(set(values)) == 1
    elif numeric == "float":
        values = [_float(row, column, context) for row in rows]
        consistent = all(
            np.isclose(
                value,
                values[0],
                rtol=CONFIG_RTOL,
                atol=CONFIG_ATOL,
            )
            for value in values[1:]
        )
    else:
        values = [row[column] for row in rows]
        consistent = len(set(values)) == 1
    if not consistent:
        raise ValueError(
            f"Inconsistent configuration metadata {column!r}: "
            f"{sorted(set(values))}"
        )


def _validate_per_catalogue_invariants(
    keyed_rows: dict[tuple[str, int, float], dict[str, str]]
) -> None:
    for scenario in SCENARIOS:
        for repop_id in sorted(EXPECTED_REPOP_IDS):
            rows = [
                keyed_rows[(scenario, repop_id, threshold)]
                for threshold in EXPECTED_THRESHOLDS
            ]
            for column in INTEGER_INVARIANT_COLUMNS:
                present = [row.get(column, "") != "" for row in rows]
                if any(present) and not all(present):
                    raise ValueError(
                        f"{column!r} is unavailable at some thresholds for "
                        f"({scenario}, repop {repop_id:04d})"
                    )
                if all(present):
                    values = [
                        _integer(
                            row,
                            column,
                            f"({scenario}, repop {repop_id:04d})",
                        )
                        for row in rows
                    ]
                    if len(set(values)) != 1:
                        raise ValueError(
                            f"Catalogue total {column!r} differs across "
                            f"thresholds for ({scenario}, repop "
                            f"{repop_id:04d}): {values}"
                        )
            for column in FLOAT_INVARIANT_COLUMNS:
                present = [row.get(column, "") != "" for row in rows]
                if any(present) and not all(present):
                    raise ValueError(
                        f"{column!r} is unavailable at some thresholds for "
                        f"({scenario}, repop {repop_id:04d})"
                    )
                if all(present):
                    values = [
                        _float(
                            row,
                            column,
                            f"({scenario}, repop {repop_id:04d})",
                        )
                        for row in rows
                    ]
                    if not all(
                        np.isclose(
                            value,
                            values[0],
                            rtol=FLOAT_RTOL,
                            atol=FLOAT_ATOL,
                        )
                        for value in values[1:]
                    ):
                        raise ValueError(
                            f"{column!r} differs across thresholds for "
                            f"({scenario}, repop {repop_id:04d}): {values}"
                        )
            for column in STRING_INVARIANT_COLUMNS:
                present = [row.get(column, "") != "" for row in rows]
                if any(present) and not all(present):
                    raise ValueError(
                        f"{column!r} is unavailable at some thresholds for "
                        f"({scenario}, repop {repop_id:04d})"
                    )
                if all(present):
                    values = [row[column] for row in rows]
                    if len(set(values)) != 1:
                        raise ValueError(
                            f"{column!r} differs across thresholds for "
                            f"({scenario}, repop {repop_id:04d}): {values}"
                        )


def combine_results(
    fragile_existing_csv: Path,
    fragile_additional_csv: Path,
    resilient_existing_csv: Path,
    resilient_additional_csv: Path,
    output_csv: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Validate four input CSVs and write the 400-row diagonal table."""
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output CSV already exists (use --overwrite): {output_csv}"
        )

    specifications = (
        ("fragile", Path(fragile_existing_csv), EXISTING_THRESHOLDS),
        ("fragile", Path(fragile_additional_csv), ADDITIONAL_THRESHOLDS),
        ("resilient", Path(resilient_existing_csv), EXISTING_THRESHOLDS),
        ("resilient", Path(resilient_additional_csv), ADDITIONAL_THRESHOLDS),
    )
    fieldnames: list[str] = []
    selected_all: list[tuple[dict[str, str], int, float]] = []

    for scenario, path, expected_thresholds in specifications:
        input_fieldnames, rows = _read_csv(path)
        for fieldname in input_fieldnames:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
        _validate_input_rows(path, rows, scenario)
        selected = _diagonal_rows(rows, path)
        _validate_expected_source_thresholds(
            selected, expected_thresholds, path
        )
        selected_all.extend(selected)

    rows_only = [row for row, _, _ in selected_all]
    has_ratio = [row.get(REFERENCE_RATIO_COLUMN, "") != "" for row in rows_only]
    has_peak = [row.get(DISCARDED_PEAK_COLUMN, "") != "" for row in rows_only]
    if not all(has_ratio) and not all(has_peak):
        raise ValueError(
            "The conservative accumulated-impact metric is not available "
            "consistently for every selected row"
        )
    metric_column = (
        REFERENCE_RATIO_COLUMN if all(has_ratio) else DISCARDED_PEAK_COLUMN
    )
    for index, row in enumerate(rows_only, start=1):
        value = _float(row, metric_column, f"selected row {index}")
        if value < 0.0:
            raise ValueError(
                f"Conservative accumulated-impact metric {metric_column!r} "
                f"is negative in selected row {index}: {value!r}"
            )

    for column in GLOBAL_INTEGER_CONFIG_COLUMNS:
        _validate_constant_column(rows_only, column, numeric="integer")
    for column in GLOBAL_FLOAT_CONFIG_COLUMNS:
        _validate_constant_column(rows_only, column, numeric="float")
    for column in GLOBAL_STRING_CONFIG_COLUMNS:
        _validate_constant_column(rows_only, column)

    keyed_rows: dict[tuple[str, int, float], dict[str, str]] = {}
    scenario_repops = {scenario: set() for scenario in SCENARIOS}
    for row, repop_id, threshold in selected_all:
        scenario = row["scenario"]
        key = (scenario, repop_id, threshold)
        if key in keyed_rows:
            raise ValueError(
                "Duplicate (scenario, repop_id, f) key: "
                f"({scenario}, {repop_id:04d}, {threshold:.0e})"
            )
        keyed_rows[key] = row
        scenario_repops[scenario].add(repop_id)

    for scenario in SCENARIOS:
        observed = scenario_repops[scenario]
        if observed != EXPECTED_REPOP_IDS:
            missing = sorted(EXPECTED_REPOP_IDS - observed)
            unexpected = sorted(observed - EXPECTED_REPOP_IDS)
            raise ValueError(
                f"Wrong repopulations for {scenario}: missing={missing}, "
                f"unexpected={unexpected}; expected 0000-0049"
            )
    if scenario_repops["fragile"] != scenario_repops["resilient"]:
        raise ValueError(
            "Fragile and resilient inputs do not contain the same "
            "repopulations"
        )

    expected_keys = {
        (scenario, repop_id, threshold)
        for scenario in SCENARIOS
        for repop_id in EXPECTED_REPOP_IDS
        for threshold in EXPECTED_THRESHOLDS
    }
    observed_keys = set(keyed_rows)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(
            "Missing or unexpected (scenario, repop_id, f) combinations: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    if len(keyed_rows) != 400:
        raise ValueError(
            f"Expected exactly 400 output rows, found {len(keyed_rows)}"
        )

    _validate_per_catalogue_invariants(keyed_rows)

    ordered_rows = [keyed_rows[key] for key in sorted(keyed_rows)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(ordered_rows)
        os.replace(temporary, output_csv)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    combine_results(
        fragile_existing_csv=args.fragile_existing_csv,
        fragile_additional_csv=args.fragile_additional_csv,
        resilient_existing_csv=args.resilient_existing_csv,
        resilient_additional_csv=args.resilient_additional_csv,
        output_csv=args.output_csv,
        overwrite=args.overwrite,
    )
    print(f"Wrote 400 validated rows: {args.output_csv}")


if __name__ == "__main__":
    main()
