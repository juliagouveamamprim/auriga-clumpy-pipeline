import hashlib
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import prepare_subhalo_components as prep


COLUMN_NAMES = [
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
]


CATALOG_DATA = np.array(
    [
        [1e19, 10.0, 20.0, 8.0, 1.0, 1.0, 1e7, 10.0, 0.0, 0.0],
        [1e16, 10.0, 20.0, 8.0, 1.0, 1.0, 1e7, -10.0, 0.0, 0.0],
        [1e20, 10.0, 20.0, 1.0, 1.0, 1.0, 1e7, 0.0, 10.0, 0.0],
        [1e14, 10.0, 20.0, 1.0, 1.0, 1.0, 1e7, 0.0, -10.0, 0.0],
    ],
    dtype=np.float64,
)


@pytest.fixture
def preparation_case(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_list = tmp_path / "extended.txt"
    output_fits = tmp_path / "pointlike.fits"

    with h5py.File(input_h5, "w") as h5:
        group = h5.create_group("iteration_0")
        dataset = group.create_dataset("data", data=CATALOG_DATA)
        dataset.attrs["column_names"] = COLUMN_NAMES
        group.create_dataset(
            "halo_name",
            data=np.array(
                [
                    b"extended_keep",
                    b"extended_drop",
                    b"pointlike_keep",
                    b"pointlike_drop",
                ]
            ),
        )

    return {
        "input_h5": input_h5,
        "output_list": output_list,
        "output_fits": output_fits,
    }


def run_preparation(case, **overrides):
    arguments = {
        "input_h5": case["input_h5"],
        "output_list": case["output_list"],
        "output_pointlike_fits": case["output_fits"],
        "scenario": "resilient",
        "repop_id": 7,
        "iteration": 0,
        "top_n": None,
        "halo_type": "DSPH",
        "nside": 8,
        "round_up_decimals": 2,
        "chunk_size": 2,
        "extended_cut_f": 1e-3,
        "pointlike_cut_f": 1e-4,
    }
    arguments.update(overrides)
    prep.prepare_subhalo_components(**arguments)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_pointlike_writes(monkeypatch):
    original = prep.write_pointlike_fits
    calls = []

    def recording_writer(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(prep, "write_pointlike_fits", recording_writer)
    return calls


def test_valid_fits_is_reused_while_extended_list_is_recreated(
    preparation_case,
    monkeypatch,
):
    run_preparation(preparation_case)

    output_fits = preparation_case["output_fits"]
    output_list = preparation_case["output_list"]
    sentinel_ns = 1_000_000_000_000_000_000
    os.utime(output_fits, ns=(sentinel_ns, sentinel_ns))
    original_checksum = sha256(output_fits)

    output_list.unlink()
    calls = record_pointlike_writes(monkeypatch)
    run_preparation(preparation_case, extended_cut_f=2e-3)

    assert output_list.is_file()
    assert "# extended_cut_f = 0.002" in output_list.read_text()
    assert calls == []
    assert output_fits.stat().st_mtime_ns == sentinel_ns
    assert sha256(output_fits) == original_checksum


def test_valid_uncut_fits_is_reused(preparation_case, monkeypatch):
    cut_arguments = {
        "extended_cut_f": None,
        "pointlike_cut_f": None,
    }
    run_preparation(preparation_case, **cut_arguments)
    calls = record_pointlike_writes(monkeypatch)

    run_preparation(preparation_case, **cut_arguments)

    assert calls == []


def test_missing_fits_is_generated(preparation_case, monkeypatch):
    calls = record_pointlike_writes(monkeypatch)

    run_preparation(preparation_case)

    assert len(calls) == 1
    assert preparation_case["output_fits"].is_file()


def test_force_pointlike_regenerates_valid_fits(preparation_case, monkeypatch):
    run_preparation(preparation_case)
    calls = record_pointlike_writes(monkeypatch)

    run_preparation(preparation_case, force_pointlike=True)

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("corruption", "value"),
    [
        ("SCENARIO", "fragile"),
        ("FPL", 2e-4),
        ("truncate", None),
    ],
)
def test_invalid_or_mismatched_metadata_regenerates_fits(
    preparation_case,
    monkeypatch,
    corruption,
    value,
):
    run_preparation(preparation_case)
    output_fits = preparation_case["output_fits"]

    if corruption == "truncate":
        output_fits.write_bytes(output_fits.read_bytes()[:100])
    else:
        with fits.open(output_fits, mode="update") as hdul:
            hdul[1].header[corruption] = value
            hdul[2].header[corruption] = value

    calls = record_pointlike_writes(monkeypatch)
    run_preparation(preparation_case)

    assert len(calls) == 1
    with fits.open(output_fits) as hdul:
        assert hdul[1].header["SCENARIO"] == "resilient"
        assert hdul[1].header["FPL"] == pytest.approx(1e-4)


def test_current_catalog_jref_mismatch_regenerates_fits(
    preparation_case,
    monkeypatch,
):
    run_preparation(preparation_case)
    output_fits = preparation_case["output_fits"]

    with fits.open(output_fits) as hdul:
        original_jref = hdul[1].header["JREF"]

    with h5py.File(preparation_case["input_h5"], "r+") as h5:
        h5["iteration_0/data"][2, 0] = 2e20

    calls = record_pointlike_writes(monkeypatch)
    run_preparation(preparation_case)

    assert len(calls) == 1
    with fits.open(output_fits) as hdul:
        assert hdul[1].header["JREF"] == pytest.approx(2e20)
        assert hdul[1].header["JREF"] != original_jref
