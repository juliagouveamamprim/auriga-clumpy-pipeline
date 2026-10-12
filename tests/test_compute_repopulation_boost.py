import csv
import importlib.util
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compute_repopulation_boost.py"
SPEC = importlib.util.spec_from_file_location("compute_repopulation_boost", SCRIPT_PATH)
boost = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = boost
SPEC.loader.exec_module(boost)


COLUMNS = ["rho_s", "Js", "r_s", "Vmax"]
UNITS = ["Msun kpc^-3", "GeV^2 cm^-5", "kpc", "km s^-1"]


def write_catalogue(
    path,
    rows,
    *,
    columns=COLUMNS,
    units=UNITS,
    n_saved=None,
    host=(2.0, 3.0, 6.0),
    delta=200.0,
    omit_host=None,
    add_renormalized=False,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("iteration_0")
        data = group.create_dataset("data", data=np.asarray(rows, dtype=np.float64))
        data.attrs["column_names"] = np.asarray(columns, dtype=h5py.string_dtype())
        data.attrs["units"] = np.asarray(units, dtype=h5py.string_dtype())
        group.attrs["n_saved"] = len(rows) if n_saved is None else n_saved

        values = {
            "rho_0": (host[0], "Msun / kpc3"),
            "r_s": (host[1], "kpc"),
            "R_vir": (host[2], "kpc"),
        }
        for name, (value, unit) in values.items():
            if omit_host == name:
                continue
            parameter = handle.create_group(f"inputs/host/{name}")
            parameter.create_dataset("value", data=value)
            parameter.create_dataset("unit", data=unit)

        if omit_host != "delta_overdensity":
            handle.create_dataset("inputs/cosmo_constants/delta_overdensity", data=delta)

        if add_renormalized:
            handle.create_dataset("inputs/host/rho0_host_new", data=1.0e-30)
            handle.create_dataset("inputs/renormalization/gMW_RHOSOL", data=1.0e30)


def source_path(root, repop_id, scenario="fragile"):
    return (
        root
        / f"repop_{repop_id:04d}"
        / f"fullrepop_hydro_{scenario}.h5"
    )


def standard_rows():
    # Column order is deliberately not the canonical HDF5 order.
    return np.array(
        [
            [1.0, 30.0, 1.0, 11.0],
            [3.0, 20.0, 1.0, 22.0],
            [2.0, 10.0, 1.0, 33.0],
        ]
    )


def test_exact_luminosity_chunk_boundaries_and_distinct_maxima(tmp_path):
    path = source_path(tmp_path, 7)
    write_catalogue(path, standard_rows())

    result, host = boost.scan_catalogue(path, repop_id=7, chunk_size=2)
    coefficient = 7.0 * math.pi / 6.0
    expected_total = coefficient * (1.0 + 9.0 + 4.0)
    expected_host = (4.0 * math.pi / 3.0) * 2.0**2 * 3.0**3 * (1.0 - 3.0**-3)

    assert result.repop_id == 7
    assert result.n_subhalos == 3
    assert float(result.l_sub_total) == pytest.approx(expected_total)
    assert float(result.boost_factor) == pytest.approx(expected_total / expected_host)
    assert float(result.sum_js) == pytest.approx(60.0)
    assert result.max_js == 30.0
    assert float(result.max_l_subhalo) == pytest.approx(9.0 * coefficient)
    assert float(result.frac_lmax) == pytest.approx(9.0 / 14.0)
    assert result.vmax_at_max_js == 11.0
    assert result.vmax_at_max_l == 22.0
    assert host == boost.HostParameters(2.0, 3.0, 6.0, 200.0)


def test_ties_keep_first_catalogue_occurrence_across_chunks(tmp_path):
    path = source_path(tmp_path, 0)
    rows = [
        [2.0, 8.0, 1.0, 10.0],
        [1.0, 2.0, 1.0, 20.0],
        [2.0, 8.0, 1.0, 30.0],
    ]
    write_catalogue(path, rows)

    result, _ = boost.scan_catalogue(path, repop_id=0, chunk_size=1)

    assert result.vmax_at_max_js == 10.0
    assert result.vmax_at_max_l == 10.0


def test_denominator_uses_original_embedded_host_and_ignores_renormalized_data(tmp_path):
    path = source_path(tmp_path, 0)
    write_catalogue(
        path,
        [[1.0, 1.0, 1.0, 1.0]],
        host=(4.0, 5.0, 20.0),
        add_renormalized=True,
    )

    result, host = boost.scan_catalogue(path, repop_id=0, chunk_size=4)
    expected_sub = 7.0 * math.pi / 6.0
    expected_host = (4.0 * math.pi / 3.0) * 4.0**2 * 5.0**3 * (1.0 - 5.0**-3)

    assert host.rho0_msun_kpc3 == 4.0
    assert float(result.boost_factor) == pytest.approx(expected_sub / expected_host)


def test_missing_required_column(tmp_path):
    path = source_path(tmp_path, 0)
    write_catalogue(
        path,
        [[1.0, 2.0, 3.0]],
        columns=["rho_s", "Js", "Vmax"],
        units=["Msun kpc^-3", "GeV^2 cm^-5", "km s^-1"],
    )

    with pytest.raises(KeyError, match="r_s"):
        boost.scan_catalogue(path, repop_id=0, chunk_size=2)


def test_column_unit_mismatch(tmp_path):
    path = source_path(tmp_path, 0)
    bad_units = list(UNITS)
    bad_units[COLUMNS.index("r_s")] = "pc"
    write_catalogue(path, standard_rows(), units=bad_units)

    with pytest.raises(ValueError, match="Unit mismatch.*r_s"):
        boost.scan_catalogue(path, repop_id=0, chunk_size=2)


def test_n_saved_must_match_dataset_rows(tmp_path):
    path = source_path(tmp_path, 0)
    write_catalogue(path, standard_rows(), n_saved=99)

    with pytest.raises(ValueError, match="Incomplete or truncated"):
        boost.scan_catalogue(path, repop_id=0, chunk_size=2)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Js", np.nan),
        ("Vmax", np.inf),
        ("r_s", 0.0),
        ("rho_s", -1.0),
    ],
)
def test_invalid_required_values_fail(tmp_path, column, value):
    path = source_path(tmp_path, 0)
    rows = standard_rows()
    rows[1, COLUMNS.index(column)] = value
    write_catalogue(path, rows)

    with pytest.raises(ValueError, match=f"Invalid {column} at row 1"):
        boost.scan_catalogue(path, repop_id=0, chunk_size=2)


@pytest.mark.parametrize("missing", ["rho_0", "r_s", "R_vir", "delta_overdensity"])
def test_missing_embedded_host_parameters_fail(tmp_path, missing):
    path = source_path(tmp_path, 0)
    write_catalogue(path, standard_rows(), omit_host=missing)

    with pytest.raises(KeyError, match="Missing required"):
        boost.scan_catalogue(path, repop_id=0, chunk_size=2)


def test_embedded_host_unit_and_overdensity_are_validated(tmp_path):
    unit_path = source_path(tmp_path / "unit", 0)
    write_catalogue(unit_path, standard_rows())
    with h5py.File(unit_path, "r+") as handle:
        del handle["inputs/host/rho_0/unit"]
        handle.create_dataset("inputs/host/rho_0/unit", data="kg / m3")
    with pytest.raises(ValueError, match="Unit mismatch"):
        boost.scan_catalogue(unit_path, repop_id=0, chunk_size=2)

    delta_path = source_path(tmp_path / "delta", 0)
    write_catalogue(delta_path, standard_rows(), delta=500.0)
    with pytest.raises(ValueError, match="delta_overdensity must be 200"):
        boost.scan_catalogue(delta_path, repop_id=0, chunk_size=2)


@pytest.mark.parametrize(
    ("location", "value"),
    [
        ("inputs/host/rho_0/value", np.nan),
        ("inputs/host/r_s/value", np.inf),
        ("inputs/host/R_vir/value", -np.inf),
        ("inputs/cosmo_constants/delta_overdensity", np.nan),
    ],
)
def test_nonfinite_embedded_host_metadata_fails(tmp_path, location, value):
    path = source_path(tmp_path, 0)
    write_catalogue(path, standard_rows())
    with h5py.File(path, "r+") as handle:
        del handle[location]
        handle.create_dataset(location, data=value)

    with pytest.raises(ValueError, match="Non-finite|must be finite"):
        boost.scan_catalogue(path, repop_id=0, chunk_size=2)


def test_run_rejects_inconsistent_hosts(tmp_path):
    root = tmp_path / "catalogues"
    write_catalogue(source_path(root, 0), standard_rows(), host=(2.0, 3.0, 6.0))
    write_catalogue(source_path(root, 1), standard_rows(), host=(2.1, 3.0, 6.0))

    output = tmp_path / "boost.csv"
    with pytest.raises(ValueError, match="inconsistent"):
        boost.run_scan(
            root=root,
            scenario="fragile",
            start_id=0,
            stop_id=2,
            chunk_size=2,
            output=output,
            resume=False,
        )
    with output.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert [int(row["repop_id"]) for row in saved] == [0]


def test_resume_has_no_duplicates_and_output_is_atomic_and_sorted(tmp_path, monkeypatch):
    root = tmp_path / "catalogues"
    for repop_id in range(3):
        write_catalogue(source_path(root, repop_id), standard_rows())

    output = tmp_path / "substructure_boost_fragile_0500.csv"
    result_1, _ = boost.scan_catalogue(source_path(root, 1), 1, 2)
    result_2, _ = boost.scan_catalogue(source_path(root, 2), 2, 2)
    # Simulate a valid but unsorted checkpoint produced elsewhere.
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=boost.CSV_FIELDS)
        writer.writeheader()
        writer.writerow(boost.result_to_row(result_2))
        writer.writerow(boost.result_to_row(result_1))

    real_replace = boost.os.replace
    replacements = []

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(boost.os, "replace", recording_replace)
    rows = boost.run_scan(
        root=root,
        scenario="fragile",
        start_id=0,
        stop_id=3,
        chunk_size=2,
        output=output,
        resume=True,
    )

    assert sorted(rows) == [0, 1, 2]
    with output.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert [int(row["repop_id"]) for row in saved] == [0, 1, 2]
    assert len({row["repop_id"] for row in saved}) == 3
    assert len(replacements) == 2
    assert all(destination == output for _, destination in replacements)
    assert all(source.parent == output.parent for source, _ in replacements)
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    # A second resume validates metadata but neither rescans nor rewrites rows.
    replacements.clear()
    boost.run_scan(
        root=root,
        scenario="fragile",
        start_id=0,
        stop_id=3,
        chunk_size=1,
        output=output,
        resume=True,
    )
    assert replacements == []


def test_resume_rejects_duplicate_repop_ids(tmp_path):
    output = tmp_path / "boost.csv"
    row = {name: "1" for name in boost.CSV_FIELDS}
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=boost.CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)

    with pytest.raises(ValueError, match="Duplicate repop_id"):
        boost.load_existing_rows(output)
