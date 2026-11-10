import importlib.util
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "cuts"
    / "scripts"
    / "scan_relative_js_cumulative.py"
)

spec = importlib.util.spec_from_file_location(
    "scan_relative_js_cumulative",
    SCRIPT_PATH,
)
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def write_catalogue(path):
    rows = np.array(
        [
            [100.0, 0.0, 12.0],
            [8.0, 0.0, 8.0],
            [4.0, 0.0, 7.0],
            [2.0, 0.0, 8.01],
            [np.nan, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("iteration_0/data", data=rows)
        dataset.attrs["column_names"] = ["Js", "unused", "Vmax"]


def test_dark_selection_is_inclusive_and_recomputes_jsmax(tmp_path):
    path = tmp_path / "catalogue.h5"
    write_catalogue(path)
    edges = np.linspace(-2.0, 0.0, 5)

    dark = scan.process_catalogue(
        path,
        chunk_size=2,
        relative_edges=edges,
        max_vmax_km_s=8.0,
    )

    assert dark["n_total"] == 2
    assert dark["n_above_vmax"] == 2
    assert dark["n_bad"] == 2
    assert dark["n_bad_vmax"] == 0
    assert dark["max_log10_js"] == pytest.approx(np.log10(8.0))
    assert dark["cdf"][-1] == 1.0


def test_omitting_vmax_cut_preserves_full_population(tmp_path):
    path = tmp_path / "catalogue.h5"
    write_catalogue(path)
    edges = np.linspace(-2.0, 0.0, 5)

    full = scan.process_catalogue(
        path,
        chunk_size=2,
        relative_edges=edges,
    )

    assert full["n_total"] == 4
    assert full["n_bad"] == 2
    assert full["n_bad_vmax"] == 0
    assert full["n_above_vmax"] == 0
    assert full["max_log10_js"] == pytest.approx(2.0)
    np.testing.assert_array_equal(
        full["cdf"],
        [0.0, 0.25, 0.75, 0.75, 1.0],
    )


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_rejects_invalid_vmax_cut(value, tmp_path):
    args = SimpleNamespace(
        repop_start=0,
        n_repops=1,
        chunk_size=1,
        max_vmax_km_s=value,
        output_npz=tmp_path / "result.npz",
    )

    with pytest.raises(ValueError, match="--max-vmax-km-s"):
        scan.validate_args(args)


def test_npz_records_applied_vmax_selection(monkeypatch, tmp_path):
    input_root = tmp_path / "catalogues"
    catalogue_dir = input_root / "repop_0000"
    catalogue_dir.mkdir(parents=True)
    write_catalogue(catalogue_dir / "fullrepop_hydro_fragile.h5")
    output_path = tmp_path / "result.npz"
    monkeypatch.setattr(
        scan,
        "parse_args",
        lambda: SimpleNamespace(
            input_root=input_root,
            output_npz=output_path,
            repop_start=0,
            n_repops=1,
            scenarios=("fragile",),
            chunk_size=2,
            max_vmax_km_s=8.0,
        ),
    )

    scan.main()

    with np.load(output_path, allow_pickle=False) as data:
        assert float(data["max_vmax_km_s"]) == 8.0
