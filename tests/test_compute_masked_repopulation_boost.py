import csv
import importlib.util
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "compute_masked_repopulation_boost.py"
SPEC = importlib.util.spec_from_file_location("compute_masked_repopulation_boost", SCRIPT_PATH)
masked = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = masked
SPEC.loader.exec_module(masked)

GLOBAL_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "compute_repopulation_boost.py"
GLOBAL_SPEC = importlib.util.spec_from_file_location(
    "compute_repopulation_boost_for_masked_tests", GLOBAL_SCRIPT_PATH
)
global_boost = importlib.util.module_from_spec(GLOBAL_SPEC)
assert GLOBAL_SPEC.loader is not None
sys.modules[GLOBAL_SPEC.name] = global_boost
GLOBAL_SPEC.loader.exec_module(global_boost)


COLUMNS = ["D_Earth", "Zearth", "r_s", "rho_s", "Js", "Vmax"]
UNITS = ["kpc", "kpc", "kpc", "Msun kpc^-3", "GeV^2 cm^-5", "km s^-1"]
HOST = (9.04e6, 20.0, 220.0)
OBSERVER = (8.5, 0.0, 0.0)


def source_path(root, repop_id, scenario="fragile"):
    return root / f"repop_{repop_id:04d}" / f"fullrepop_hydro_{scenario}.h5"


def write_catalogue(
    path,
    rows,
    *,
    columns=COLUMNS,
    units=UNITS,
    host=HOST,
    observer=OBSERVER,
    n_saved=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("iteration_0")
        data = group.create_dataset("data", data=np.asarray(rows, dtype=np.float64))
        data.attrs["column_names"] = np.asarray(columns, dtype=h5py.string_dtype())
        data.attrs["units"] = np.asarray(units, dtype=h5py.string_dtype())
        group.attrs["n_saved"] = len(rows) if n_saved is None else n_saved

        for name, value, unit in (
            ("rho_0", host[0], "Msun / kpc3"),
            ("r_s", host[1], "kpc"),
            ("R_vir", host[2], "kpc"),
        ):
            parameter = handle.create_group(f"inputs/host/{name}")
            parameter.create_dataset("value", data=value)
            parameter.create_dataset("unit", data=unit)
        handle.create_dataset("inputs/cosmo_constants/delta_overdensity", data=200.0)

        position = handle.create_group("inputs/host/position_Earth")
        position.create_dataset("unit", data="kpc")
        values = position.create_group("value")
        for index, value in enumerate(observer):
            values.create_dataset(f"item_{index}", data=value)


def selection_rows():
    boundary = math.sin(math.radians(10.0))
    return np.asarray(
        [
            # D, Z, rs, rho_s, Js, Vmax
            [1.0, 0.0, 1.0, 1.0, 10.0, 10.0],  # masked
            [1.0, 0.5, 2.0, 1.0, 20.0, 20.0],  # unmasked
            [1.0, boundary, 1.0, 2.0, 30.0, 30.0],  # boundary: preserved
            [1.0, -0.5, 1.0, 3.0, 40.0, 40.0],  # unmasked
        ]
    )


def scan_with_total_denominator(path, chunk_size, b_cut=10.0):
    host = masked.HostParameters(HOST[0], HOST[1], HOST[2], 200.0)
    return masked.scan_catalogue(
        path,
        repop_id=0,
        chunk_size=chunk_size,
        mask_b_cut_deg=b_cut,
        l_mw_unmasked=masked.host_luminosity(host),
    )[0]


def test_numerator_mask_selection_and_boundary(tmp_path):
    path = source_path(tmp_path, 0)
    write_catalogue(path, selection_rows())

    result = scan_with_total_denominator(path, chunk_size=2)
    coefficient = 7.0 * math.pi / 6.0

    assert result.n_subhalos_total == 4
    assert result.n_subhalos_unmasked == 3
    assert float(result.l_sub_total) == pytest.approx(coefficient * 22.0)
    assert float(result.l_sub_unmasked) == pytest.approx(coefficient * 21.0)
    assert 0 <= result.n_subhalos_unmasked <= result.n_subhalos_total
    assert 0.0 <= result.l_sub_unmasked <= result.l_sub_total


def test_full_sky_numerator_matches_global_script(tmp_path):
    path = source_path(tmp_path, 0)
    write_catalogue(path, selection_rows())

    masked_result = scan_with_total_denominator(path, chunk_size=3, b_cut=0.0)
    global_result, _ = global_boost.scan_catalogue(path, repop_id=0, chunk_size=2)

    assert masked_result.n_subhalos_unmasked == masked_result.n_subhalos_total
    assert masked_result.l_sub_unmasked == masked_result.l_sub_total
    assert float(masked_result.l_sub_total) == pytest.approx(float(global_result.l_sub_total))
    assert float(masked_result.boost_factor_global) == pytest.approx(
        float(global_result.boost_factor)
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 10])
def test_numerator_is_independent_of_chunk_size(tmp_path, chunk_size):
    path = source_path(tmp_path, 0)
    write_catalogue(path, selection_rows())

    result = scan_with_total_denominator(path, chunk_size=chunk_size)
    coefficient = 7.0 * math.pi / 6.0
    assert result.n_subhalos_unmasked == 3
    assert float(result.l_sub_unmasked) == pytest.approx(coefficient * 21.0)


def test_required_units_and_complete_row_count_are_validated(tmp_path):
    bad_unit_path = source_path(tmp_path / "unit", 0)
    bad_units = list(UNITS)
    bad_units[COLUMNS.index("Zearth")] = "pc"
    write_catalogue(bad_unit_path, selection_rows(), units=bad_units)
    with pytest.raises(ValueError, match="Unit mismatch.*Zearth"):
        masked.inspect_catalogue(bad_unit_path)

    incomplete_path = source_path(tmp_path / "incomplete", 0)
    write_catalogue(incomplete_path, selection_rows(), n_saved=99)
    with pytest.raises(ValueError, match="Incomplete or truncated"):
        masked.inspect_catalogue(incomplete_path)


def test_observer_geometry_is_read_and_validated(tmp_path):
    path = source_path(tmp_path, 0)
    write_catalogue(path, selection_rows())
    host, observer = masked.inspect_catalogue(path)
    assert host == masked.HostParameters(HOST[0], HOST[1], HOST[2], 200.0)
    np.testing.assert_allclose(observer, OBSERVER)

    bad_path = source_path(tmp_path / "bad", 0)
    write_catalogue(bad_path, selection_rows(), observer=(8.5, 0.0, 0.1))
    with pytest.raises(ValueError, match=r"position_Earth=\(R0, 0, 0\)"):
        masked.inspect_catalogue(bad_path)


def test_smooth_full_sky_quadrature_matches_analytic_host_luminosity():
    host = masked.HostParameters(HOST[0], HOST[1], HOST[2], 200.0)
    numerical = masked.smooth_host_luminosity_region(
        host, OBSERVER[0], 0.0, rtol=1.0e-10
    )
    analytic = masked.host_luminosity(host)
    assert float(numerical) == pytest.approx(float(analytic), rel=2.0e-10)


def test_smooth_mask_conservation_and_physical_bounds():
    host = masked.HostParameters(HOST[0], HOST[1], HOST[2], 200.0)
    unmasked = masked.smooth_host_luminosity_region(
        host, OBSERVER[0], 10.0, keep_unmasked=True, rtol=1.0e-8
    )
    masked_band = masked.smooth_host_luminosity_region(
        host, OBSERVER[0], 10.0, keep_unmasked=False, rtol=1.0e-8
    )
    total = masked.host_luminosity(host)

    assert 0.0 < unmasked < total
    assert 0.0 < masked_band < total
    assert float(unmasked + masked_band) == pytest.approx(float(total), rel=2.0e-8)


def test_smooth_quadrature_converges_with_tighter_tolerances():
    host = masked.HostParameters(HOST[0], HOST[1], HOST[2], 200.0)
    coarse = masked.smooth_host_luminosity_region(
        host, OBSERVER[0], 10.0, rtol=1.0e-4
    )
    medium = masked.smooth_host_luminosity_region(
        host, OBSERVER[0], 10.0, rtol=1.0e-6
    )
    fine = masked.smooth_host_luminosity_region(
        host, OBSERVER[0], 10.0, rtol=1.0e-8
    )

    assert abs(medium - fine) < abs(coarse - fine)
    assert float(abs(medium - fine) / fine) < 1.0e-6


def test_smooth_mask_matches_independent_observer_centred_los_quadrature():
    """Cross-check the Galactocentric integral with explicit s^2 ds dOmega."""
    host = masked.HostParameters(HOST[0], HOST[1], HOST[2], 200.0)
    expected = masked.smooth_host_luminosity_region(
        host, OBSERVER[0], 10.0, rtol=1.0e-8
    )

    n_b, n_l, n_s = 96, 192, 192
    x_b, w_b = np.polynomial.legendre.leggauss(n_b)
    x_l, w_l = np.polynomial.legendre.leggauss(n_l)
    x_s, w_s = np.polynomial.legendre.leggauss(n_s)
    b_min = math.radians(10.0)
    b = 0.5 * (math.pi / 2.0 - b_min) * x_b + 0.5 * (math.pi / 2.0 + b_min)
    w_b = 0.5 * (math.pi / 2.0 - b_min) * w_b
    longitude = math.pi * (x_l + 1.0)
    w_l = math.pi * w_l

    total = 0.0
    for latitude, latitude_weight in zip(b, w_b):
        cos_b = math.cos(latitude)
        mu = cos_b * np.cos(longitude)
        s_max = OBSERVER[0] * mu + np.sqrt(
            HOST[2] ** 2 - OBSERVER[0] ** 2 * (1.0 - mu**2)
        )
        distance = 0.5 * s_max[:, None] * (x_s[None, :] + 1.0)
        distance_weight = 0.5 * s_max[:, None] * w_s[None, :]
        radius = np.sqrt(
            OBSERVER[0] ** 2
            + distance**2
            - 2.0 * OBSERVER[0] * distance * mu[:, None]
        )
        x_nfw = radius / HOST[1]
        density = HOST[0] / (x_nfw * (1.0 + x_nfw) ** 2)
        los = np.sum(distance_weight * density**2 * distance**2, axis=1)
        total += 2.0 * latitude_weight * cos_b * np.sum(w_l * los)

    assert total == pytest.approx(float(expected), rel=1.0e-7)


def test_resume_keeps_one_sorted_row_per_repopulation(tmp_path):
    root = tmp_path / "catalogues"
    write_catalogue(source_path(root, 0), selection_rows())
    write_catalogue(source_path(root, 1), selection_rows())
    output = tmp_path / "masked.csv"

    masked.run_scan(
        root=root,
        scenario="fragile",
        start_id=0,
        stop_id=2,
        chunk_size=2,
        mask_b_cut_deg=0.0,
        smooth_rtol=1.0e-8,
        output=output,
        resume=False,
    )
    masked.run_scan(
        root=root,
        scenario="fragile",
        start_id=0,
        stop_id=2,
        chunk_size=1,
        mask_b_cut_deg=0.0,
        smooth_rtol=1.0e-8,
        output=output,
        resume=True,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["repop_id"]) for row in rows] == [0, 1]
    assert all(float(row["mask_b_cut_deg"]) == 0.0 for row in rows)
