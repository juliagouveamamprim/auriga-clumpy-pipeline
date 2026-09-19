import csv
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "catalogue"
    / "scripts"
    / "plot_dmin_vmax_distribution.py"
)

spec = importlib.util.spec_from_file_location(
    "plot_dmin_vmax_distribution",
    SCRIPT_PATH,
)
dmin_vmax = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dmin_vmax)


def make_catalogue(path: Path, dmin_kpc: float, vmax_kms: float) -> None:
    """Create the smallest HDF5 catalogue needed by the chunked scanner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    earth_position = np.asarray([8.0, 0.0, 0.0])
    data = np.asarray(
        [
            [vmax_kms, earth_position[0] - dmin_kpc, 0.0, 0.0],
            [0.9, earth_position[0] - (dmin_kpc + 1.0), 0.0, 0.0],
        ]
    )
    with h5py.File(path, "w") as handle:
        catalogue = handle.create_dataset("iteration_0/data", data=data)
        catalogue.attrs["column_names"] = np.asarray(
            ["Vmax", "Xearth", "Yearth", "Zearth"], dtype="S"
        )
        position = handle.create_group("inputs/host/position_Earth")
        position.create_dataset("unit", data=np.bytes_("kpc"))
        values = position.create_group("value")
        for component, value in enumerate(earth_position):
            values.create_dataset(f"item_{component}", data=value)


def write_reference_csv(path: Path, dmins: dict[int, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario", "repop_id", "min_dgc_kpc"],
        )
        writer.writeheader()
        for repop_id, dmin in sorted(dmins.items()):
            writer.writerow(
                {
                    "scenario": "resilient",
                    "repop_id": repop_id,
                    "min_dgc_kpc": dmin,
                }
            )


def run_cli(monkeypatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), *arguments])
    dmin_vmax.main()


def test_checkpoint_then_resume_without_reopening_completed_hdf5(
    tmp_path,
    monkeypatch,
):
    input_root = tmp_path / "catalogues"
    first_catalogue = input_root / "repop_0000" / "fullrepop_hydro_resilient.h5"
    second_catalogue = input_root / "repop_0001" / "fullrepop_hydro_resilient.h5"
    make_catalogue(first_catalogue, dmin_kpc=1.0, vmax_kms=0.11)
    make_catalogue(second_catalogue, dmin_kpc=2.0, vmax_kms=0.22)
    reference_csv = tmp_path / "reference.csv"
    output_csv = tmp_path / "checkpoint.csv"
    write_reference_csv(reference_csv, {0: 1.0, 1: 2.0})

    run_cli(
        monkeypatch,
        "--input-root",
        str(input_root),
        "--input-csv",
        str(reference_csv),
        "--output-csv",
        str(output_csv),
        "--n-repops",
        "1",
        "--skip-plot",
    )
    checkpoint = dmin_vmax.load_output_records(output_csv)
    assert set(checkpoint) == {("resilient", 0)}
    assert checkpoint[("resilient", 0)]["source_h5"] == str(
        first_catalogue.resolve()
    )

    scanned_paths = []
    original_scanner = dmin_vmax.find_dmin_subhalo_vmax

    def record_scanned_path(path, chunk_size):
        scanned_paths.append(path)
        return original_scanner(path, chunk_size)

    monkeypatch.setattr(
        dmin_vmax,
        "find_dmin_subhalo_vmax",
        record_scanned_path,
    )
    run_cli(
        monkeypatch,
        "--input-root",
        str(input_root),
        "--input-csv",
        str(reference_csv),
        "--output-csv",
        str(output_csv),
        "--n-repops",
        "2",
        "--skip-plot",
    )

    assert scanned_paths == [second_catalogue]
    checkpoint = dmin_vmax.load_output_records(output_csv)
    assert set(checkpoint) == {("resilient", 0), ("resilient", 1)}
    assert checkpoint[("resilient", 1)]["dmin_kpc"] == 2.0
    assert checkpoint[("resilient", 1)]["vmax_kms"] == 0.22


def test_checkpoint_rejects_duplicate_keys_and_invalid_schema(tmp_path):
    duplicate_csv = tmp_path / "duplicate.csv"
    row = {
        "scenario": "resilient",
        "repop_id": "0",
        "source_h5": "/lattes/repop_0000/fullrepop_hydro_resilient.h5",
        "dmin_kpc": "1.0",
        "vmax_kms": "0.11",
    }
    with duplicate_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=dmin_vmax.OUTPUT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)
    with pytest.raises(ValueError, match="Duplicate checkpoint key"):
        dmin_vmax.load_output_records(duplicate_csv)

    invalid_schema_csv = tmp_path / "invalid-schema.csv"
    invalid_schema_csv.write_text("scenario,repop_id,dmin_kpc\nresilient,0,1\n")
    with pytest.raises(ValueError, match="schema"):
        dmin_vmax.load_output_records(invalid_schema_csv)


def test_skip_plot_does_not_call_plotter(tmp_path, monkeypatch):
    input_root = tmp_path / "catalogues"
    catalogue = input_root / "repop_0000" / "fullrepop_hydro_resilient.h5"
    make_catalogue(catalogue, dmin_kpc=1.0, vmax_kms=0.11)
    reference_csv = tmp_path / "reference.csv"
    write_reference_csv(reference_csv, {0: 1.0})

    def savefig_must_not_be_reached(*args, **kwargs):
        raise AssertionError("--skip-plot must not call the plotting function")

    monkeypatch.setattr(
        dmin_vmax,
        "plot_vmax_distribution",
        savefig_must_not_be_reached,
    )
    run_cli(
        monkeypatch,
        "--input-root",
        str(input_root),
        "--input-csv",
        str(reference_csv),
        "--output-csv",
        str(tmp_path / "checkpoint.csv"),
        "--n-repops",
        "1",
        "--skip-plot",
    )


def test_plot_only_never_opens_hdf5_or_reference_csv(tmp_path, monkeypatch):
    output_csv = tmp_path / "checkpoint.csv"
    dmin_vmax.write_output_records_atomic(
        output_csv,
        {
            ("resilient", 0): {
                "scenario": "resilient",
                "repop_id": 0,
                "source_h5": "/lattes/repop_0000/fullrepop_hydro_resilient.h5",
                "dmin_kpc": 1.0,
                "vmax_kms": 0.11,
            }
        },
    )
    observed = {}

    def hdf5_must_not_be_opened(*args, **kwargs):
        raise AssertionError("--plot-only must not open HDF5")

    def capture_plot(values, scenario, output_dir, n_bins, dpi):
        observed["values"] = values
        observed["scenario"] = scenario

    monkeypatch.setattr(dmin_vmax.h5py, "File", hdf5_must_not_be_opened)
    monkeypatch.setattr(dmin_vmax, "plot_vmax_distribution", capture_plot)
    run_cli(
        monkeypatch,
        "--input-root",
        str(tmp_path / "missing-hdf5"),
        "--input-csv",
        str(tmp_path / "missing-reference.csv"),
        "--output-csv",
        str(output_csv),
        "--n-repops",
        "1",
        "--plot-only",
    )

    assert observed["scenario"] == "resilient"
    assert np.array_equal(observed["values"], np.asarray([0.11]))
