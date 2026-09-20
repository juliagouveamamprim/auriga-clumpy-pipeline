import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import healpy as hp
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

SYNTHETIC_GMW_RHOSOL = 0.4


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json_loads(text):
    def reject_nonfinite(token):
        raise ValueError(f"non-finite JSON token: {token}")

    return json.loads(text, parse_constant=reject_nonfinite)


def make_case_inputs(
    repository_root,
    scenario="resilient",
    template_gmw=None,
    discarded_js=1e14,
):
    input_h5 = (
        repository_root
        / "outputs"
        / "repop_0007"
        / f"fullrepop_hydro_{scenario}.h5"
    )
    input_h5.parent.mkdir(parents=True, exist_ok=True)

    data = np.array(
        [
            [1e19, 10.0, 20.0, 8.0, 1.0, 1.0, 1e7, 10.0, 0.0, 0.0],
            [discarded_js, 10.0, 20.0, 8.0, 1.0, 1.0, 1e7, -10.0, 0.0, 0.0],
            [1e20, 10.0, 20.0, 1.0, 1.0, 1.0, 1e7, 0.0, 10.0, 0.0],
            [discarded_js, 10.0, 20.0, 1.0, 1.0, 1.0, 1e7, 0.0, -10.0, 0.0],
            [np.nan, 10.0, 20.0, 1.0, 1.0, 1.0, 1e7, 0.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    names = np.array(
        [
            b"extended_keep",
            b"extended_drop",
            b"pointlike_keep",
            b"pointlike_drop",
            b"invalid",
        ]
    )

    with h5py.File(input_h5, "w") as h5:
        group = h5.create_group("iteration_0")
        group.attrs["n_generated"] = 8
        group.attrs["n_removed_engulfing"] = 2
        group.attrs["n_removed_roche"] = 1
        group.attrs["n_saved"] = 5
        group.attrs["unused_nonfinite_attribute"] = np.nan
        dataset = group.create_dataset("data", data=data)
        dataset.attrs["column_names"] = COLUMN_NAMES
        dataset.attrs["units"] = ["test-unit"] * len(COLUMN_NAMES)
        group.create_dataset("halo_name", data=names)

    template = (
        repository_root
        / "configs"
        / "clumpy_templates"
        / f"clumpy_params_g6_auriga_nfw_{scenario}_renorm_vmin0p1.template.txt"
    )
    template.parent.mkdir(parents=True, exist_ok=True)
    if template_gmw is None:
        template_gmw = SYNTHETIC_GMW_RHOSOL
    template.write_text(
        "# Synthetic scientific template\n"
        f"# Scenario: hydro_{scenario}\n"
        f"gMW_RHOSOL [GeV/cm3] {template_gmw:.10e} <float> fixed\n"
        "gSIM_HEALPIX_NSIDE [-] 8 <integer> test\n",
        encoding="utf-8",
    )

    return input_h5, template


def test_cli_defaults_and_explicit_no_cuts():
    args = prep.parse_args(["7", "resilient"])
    assert args.nside == 2048
    assert args.extended_cut_f == pytest.approx(1e-3)
    assert args.pointlike_cut_f == pytest.approx(1e-3)

    args = prep.parse_args(["7", "resilient", "--cut-f", "2e-4"])
    assert args.extended_cut_f == pytest.approx(2e-4)
    assert args.pointlike_cut_f == pytest.approx(2e-4)

    args = prep.parse_args(["7", "resilient", "--no-cuts"])
    assert args.extended_cut_f is None
    assert args.pointlike_cut_f is None

    args = prep.parse_args(
        [
            "7",
            "resilient",
            "--extended-cut-f",
            "2e-3",
            "--pointlike-cut-f",
            "4e-4",
        ]
    )
    assert args.extended_cut_f == pytest.approx(2e-3)
    assert args.pointlike_cut_f == pytest.approx(4e-4)

    with pytest.raises(SystemExit):
        prep.parse_args(
            ["7", "resilient", "--no-cuts", "--extended-cut-f", "1e-3"]
        )

    with pytest.raises(SystemExit):
        prep.parse_args(
            ["7", "resilient", "--cut-f", "1e-3", "--pointlike-cut-f", "2e-3"]
        )


@pytest.mark.parametrize(
    "flag",
    ["--cut-f", "--extended-cut-f", "--pointlike-cut-f"],
)
@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
def test_cli_rejects_nonfinite_cut_factors(flag, token):
    with pytest.raises(SystemExit):
        prep.parse_args(["7", "resilient", flag, token])


def test_manifest_writer_rejects_nonfinite_json(tmp_path):
    target = tmp_path / "manifest.json"

    with pytest.raises(ValueError):
        prep.write_json_atomic_no_overwrite(target, {"value": float("nan")})

    assert not target.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("scenario", ("resilient", "fragile"))
def test_official_template_declares_scenario_and_finite_gmw(scenario):
    template_path = prep.get_template_path(scenario)
    template_bytes = template_path.read_bytes()

    assert prep.read_template_scenario_from_bytes(template_bytes) == scenario
    assert np.isfinite(
        prep.read_gmw_rhosol_from_bytes(
            template_bytes,
            source_description=str(template_path),
        )
    )


@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
def test_rejects_nonfinite_gmw_rhosol(token):
    template_bytes = (
        f"gMW_RHOSOL [GeV/cm3] {token} <float> fixed\n".encode("utf-8")
    )

    with pytest.raises(ValueError, match="must be finite"):
        prep.read_gmw_rhosol_from_bytes(template_bytes)


def test_prepare_case_rejects_template_from_other_scenario(tmp_path):
    make_case_inputs(tmp_path, scenario="resilient")
    _, fragile_template = make_case_inputs(tmp_path, scenario="fragile")

    with pytest.raises(ValueError, match="does not match scenario"):
        prep.prepare_case(
            repop_id=7,
            scenario="resilient",
            nside=8,
            base_h5_dir=tmp_path / "outputs",
            base_run_dir=tmp_path / "outputs" / "clumpy",
            template_path=fragile_template,
            chunk_size=2,
        )

    case_dir = prep.get_case_dir(
        7,
        "resilient",
        8,
        base_run_dir=tmp_path / "outputs" / "clumpy",
    )
    assert not case_dir.exists()


@pytest.mark.parametrize(
    "template_text",
    [
        "gMW_RHOSOL [GeV/cm3] 0.4 <float> fixed\n",
        (
            "# Scenario: hydro_fragile\n"
            "# Scenario: hydro_resilient\n"
            "gMW_RHOSOL [GeV/cm3] 0.4 <float> fixed\n"
        ),
    ],
)
def test_template_scenario_marker_must_be_unique(template_text):
    with pytest.raises(ValueError, match="Expected exactly one"):
        prep.validate_template_scenario(
            template_text.encode("utf-8"),
            "resilient",
            "synthetic template",
        )


@pytest.mark.parametrize("scenario", ["resilient", "fragile"])
def test_prepare_case_writes_complete_relative_manifest(tmp_path, scenario):
    input_h5, template = make_case_inputs(tmp_path, scenario=scenario)
    hdf5_checksum_before = checksum(input_h5)

    manifest_path, manifest = prep.prepare_case(
        repop_id=7,
        scenario=scenario,
        nside=8,
        base_h5_dir=tmp_path / "outputs",
        base_run_dir=tmp_path / "outputs" / "clumpy",
        template_path=template,
        repository_root=tmp_path,
        chunk_size=2,
    )

    manifest_base = manifest_path.parent / manifest["paths"]["base"]

    def resolve(record):
        return (manifest_base / record["path"]).resolve()

    raw_list = resolve(manifest["artifacts"]["raw_list"])
    pointlike = resolve(manifest["artifacts"]["pointlike_fits"])
    snapshot = resolve(manifest["artifacts"]["template_snapshot"])

    assert manifest_path.exists()
    assert strict_json_loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["schema_version"] == 1
    assert manifest["state"] == "complete"
    assert manifest["paths"]["base"] == "."
    assert manifest["identity"] == {
        "repop_id": 7,
        "repop_tag": "repop_0007",
        "scenario": scenario,
        "nside": 8,
        "hdf5_iteration": 0,
    }

    cuts = manifest["scientific_configuration"]["cuts"]
    assert cuts["enabled"] is True
    assert cuts["extended_cut_f"] == pytest.approx(1e-3)
    assert cuts["pointlike_cut_f"] == pytest.approx(1e-3)
    assert cuts["j_pixel_ref"] == pytest.approx(1e20)
    assert cuts["extended_j_cut"] == pytest.approx(1e17)
    assert cuts["pointlike_j_cut"] == pytest.approx(1e17)
    assert manifest["scientific_configuration"]["smooth_milky_way"][
        "gMW_RHOSOL_GeV_cm3"
    ] == pytest.approx(SYNTHETIC_GMW_RHOSOL)

    for source_name, expected_path in (
        ("hdf5", input_h5),
        ("template", template),
    ):
        source = manifest["source"][source_name]
        assert not Path(source["path"]).is_absolute()
        assert resolve(source) == expected_path.resolve()
        assert source["provenance_only"] is True
        assert source["required_by_rendering"] is False

    hdf5_stats = manifest["catalogue_statistics"]["hdf5"]
    assert hdf5_stats["rows"] == 5
    assert hdf5_stats["group_attributes"] == {
        "n_generated": 8,
        "n_removed_engulfing": 2,
        "n_removed_roche": 1,
        "n_saved": 5,
    }
    assert hdf5_stats["data_attributes"] == {"column_names": COLUMN_NAMES}

    stats = manifest["catalogue_statistics"]["preparation"]
    assert stats["rows_seen"] == 5
    assert stats["valid_count"] == 4
    assert stats["invalid_count"] == 1
    assert stats["extended"]["before_cuts_count"] == 2
    assert stats["extended"]["after_cuts_count"] == 1
    assert stats["extended"]["discarded_count"] == 1
    assert stats["extended"]["written_count"] == 1
    assert stats["extended"]["discarded_js_sum"] == pytest.approx(1e14)
    assert stats["pointlike"]["before_cuts_count"] == 2
    assert stats["pointlike"]["after_cuts_count"] == 1
    assert stats["pointlike"]["discarded_count"] == 1
    assert stats["pointlike"]["discarded_js_sum"] == pytest.approx(1e14)
    assert stats["pointlike"]["map_js_sum"] == pytest.approx(1e20)

    for artifact_name, artifact_path in (
        ("raw_list", raw_list),
        ("pointlike_fits", pointlike),
        ("template_snapshot", snapshot),
    ):
        artifact = manifest["artifacts"][artifact_name]
        assert not Path(artifact["path"]).is_absolute()
        assert artifact["sha256"] == checksum(artifact_path)
        assert artifact["size_bytes"] == artifact_path.stat().st_size

    rows = [
        line
        for line in raw_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rows == [
        "extended_keep   DSPH       0.000000   0.000000    10.000000 "
        "    -1     1.000000   2.50000000e+06   1.000000 kZHAO       "
        "1    3    1"
    ]

    with fits.open(pointlike) as hdul:
        assert hdul[1].name == "JFACTOR"
        assert hdul[2].name == "JFACTOR_PER_SR"
        assert hdul[1].header["ORDERING"] == "NESTED"
        assert hdul[2].header["ORDERING"] == "NESTED"
        assert hdul[1].header["NSIDE"] == 8
        assert hdul[1].columns["Jpointlike"].unit == "GeV^2 cm^-5"
        assert (
            hdul[2].columns["Jpointlike_per_sr"].unit
            == "GeV^2 cm^-5 sr^-1"
        )
        assert hdul[1].columns["Jpointlike"].format == "1D"
        assert hdul[2].columns["Jpointlike_per_sr"].format == "1D"

        jmap = np.asarray(hdul[1].data["Jpointlike"], dtype=np.float64)
        jsr = np.asarray(
            hdul[2].data["Jpointlike_per_sr"],
            dtype=np.float64,
        )
        pixel = hp.ang2pix(8, 90.0, 0.0, lonlat=True, nest=True)
        assert jmap.sum() == pytest.approx(1e20)
        assert jmap[pixel] == pytest.approx(1e20)
        assert np.allclose(jsr * hp.nside2pixarea(8), jmap, rtol=1e-12)

    assert snapshot.read_bytes() == template.read_bytes()
    assert checksum(input_h5) == hdf5_checksum_before

    scenario_dir = tmp_path / "outputs" / "clumpy" / scenario
    assert not (scenario_dir / "params").exists()
    assert not (scenario_dir / "outputs").exists()
    assert not (scenario_dir / "lists" / "corrected").exists()
    assert not (scenario_dir / "validation").exists()
    assert not (manifest_path.parent / "rendering_record.json").exists()
    assert not os.path.lexists(manifest_path.parent / ".preparation.lock")
    assert list(manifest_path.parent.glob(".*.tmp")) == []


def test_discarded_js_sums_are_accumulated_directly(tmp_path):
    _, template = make_case_inputs(tmp_path, discarded_js=1.0)

    _, manifest = prep.prepare_case(
        repop_id=7,
        scenario="resilient",
        nside=8,
        base_h5_dir=tmp_path / "outputs",
        base_run_dir=tmp_path / "outputs" / "clumpy",
        template_path=template,
        chunk_size=1,
    )

    stats = manifest["catalogue_statistics"]["preparation"]
    assert stats["extended"]["discarded_js_sum"] == 1.0
    assert stats["pointlike"]["discarded_js_sum"] == 1.0


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_known_hdf5_counter_must_be_finite_integer(tmp_path, nonfinite):
    input_h5, template = make_case_inputs(tmp_path)
    with h5py.File(input_h5, "r+") as h5:
        h5["iteration_0"].attrs["n_saved"] = nonfinite

    with pytest.raises(ValueError, match="n_saved"):
        prep.prepare_case(
            repop_id=7,
            scenario="resilient",
            nside=8,
            base_h5_dir=tmp_path / "outputs",
            base_run_dir=tmp_path / "outputs" / "clumpy",
            template_path=template,
            chunk_size=2,
        )

    case_dir = prep.get_case_dir(
        7,
        "resilient",
        8,
        base_run_dir=tmp_path / "outputs" / "clumpy",
    )
    assert not (case_dir / "preparation_manifest.json").exists()
    assert not os.path.lexists(case_dir / ".preparation.lock")


def test_nonfinite_scientific_sum_prevents_manifest_publication(tmp_path):
    input_h5, template = make_case_inputs(tmp_path)
    with h5py.File(input_h5, "r+") as h5:
        h5["iteration_0/data"][0, 0] = 1e308
        h5["iteration_0/data"][1, 0] = 1e308

    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="Non-finite scientific value"):
            prep.prepare_case(
                repop_id=7,
                scenario="resilient",
                nside=8,
                base_h5_dir=tmp_path / "outputs",
                base_run_dir=tmp_path / "outputs" / "clumpy",
                template_path=template,
                chunk_size=2,
            )

    case_dir = prep.get_case_dir(
        7,
        "resilient",
        8,
        base_run_dir=tmp_path / "outputs" / "clumpy",
    )
    assert not (case_dir / "preparation_manifest.json").exists()
    assert not os.path.lexists(case_dir / ".preparation.lock")


def test_prepare_case_refuses_all_existing_targets_before_writing(tmp_path):
    _, template = make_case_inputs(tmp_path)
    base_run_dir = tmp_path / "outputs" / "clumpy"
    raw_list = prep.get_output_list(
        7,
        "resilient",
        None,
        nside=8,
        base_run_dir=base_run_dir,
    )
    pointlike = prep.get_output_pointlike_fits(
        7,
        "resilient",
        8,
        base_run_dir=base_run_dir,
    )
    case_dir = prep.get_case_dir(
        7,
        "resilient",
        8,
        base_run_dir=base_run_dir,
    )
    snapshot = case_dir / "clumpy_params.template.txt"

    raw_list.parent.mkdir(parents=True)
    pointlike.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    raw_list.write_text("existing raw\n", encoding="utf-8")
    pointlike.write_bytes(b"existing fits")
    snapshot.write_text("existing template\n", encoding="utf-8")

    with pytest.raises(FileExistsError) as exc_info:
        prep.prepare_case(
            repop_id=7,
            scenario="resilient",
            nside=8,
            base_h5_dir=tmp_path / "outputs",
            base_run_dir=base_run_dir,
            template_path=template,
            repository_root=tmp_path,
        )

    message = str(exc_info.value)
    assert str(raw_list) in message
    assert str(pointlike) in message
    assert str(snapshot) in message
    assert "--force is not implemented yet" in message
    assert raw_list.read_text(encoding="utf-8") == "existing raw\n"
    assert pointlike.read_bytes() == b"existing fits"
    assert snapshot.read_text(encoding="utf-8") == "existing template\n"
    assert not (case_dir / "preparation_manifest.json").exists()


def test_prepare_case_refuses_broken_symlink_target_before_writing(tmp_path):
    _, template = make_case_inputs(tmp_path)
    base_run_dir = tmp_path / "outputs" / "clumpy"
    raw_list = prep.get_output_list(
        7,
        "resilient",
        None,
        nside=8,
        base_run_dir=base_run_dir,
    )
    raw_list.parent.mkdir(parents=True)
    raw_list.symlink_to(tmp_path / "missing-symlink-target")

    with pytest.raises(FileExistsError) as exc_info:
        prep.prepare_case(
            repop_id=7,
            scenario="resilient",
            nside=8,
            base_h5_dir=tmp_path / "outputs",
            base_run_dir=base_run_dir,
            template_path=template,
        )

    assert str(raw_list) in str(exc_info.value)
    assert os.path.lexists(raw_list)
    case_dir = prep.get_case_dir(7, "resilient", 8, base_run_dir=base_run_dir)
    assert not os.path.lexists(case_dir / ".preparation.lock")
    assert not (case_dir / "preparation_manifest.json").exists()


def test_prepare_case_refuses_concurrent_case_lock(tmp_path):
    _, template = make_case_inputs(tmp_path)
    base_run_dir = tmp_path / "outputs" / "clumpy"
    case_dir = prep.get_case_dir(7, "resilient", 8, base_run_dir=base_run_dir)
    lock_path = case_dir / ".preparation.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("another preparation\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already locked"):
        prep.prepare_case(
            repop_id=7,
            scenario="resilient",
            nside=8,
            base_h5_dir=tmp_path / "outputs",
            base_run_dir=base_run_dir,
            template_path=template,
        )

    assert lock_path.read_text(encoding="utf-8") == "another preparation\n"
    assert not (case_dir / "preparation_manifest.json").exists()


def test_failure_before_manifest_keeps_products_and_removes_lock(
    tmp_path,
    monkeypatch,
):
    _, template = make_case_inputs(tmp_path)
    base_run_dir = tmp_path / "outputs" / "clumpy"

    def fail_manifest_publication(path, payload):
        raise RuntimeError("simulated manifest failure")

    monkeypatch.setattr(
        prep,
        "write_json_atomic_no_overwrite",
        fail_manifest_publication,
    )

    with pytest.raises(RuntimeError, match="simulated manifest failure"):
        prep.prepare_case(
            repop_id=7,
            scenario="resilient",
            nside=8,
            base_h5_dir=tmp_path / "outputs",
            base_run_dir=base_run_dir,
            template_path=template,
            chunk_size=2,
        )

    raw_list = prep.get_output_list(
        7,
        "resilient",
        None,
        nside=8,
        base_run_dir=base_run_dir,
    )
    pointlike = prep.get_output_pointlike_fits(
        7,
        "resilient",
        8,
        base_run_dir=base_run_dir,
    )
    case_dir = prep.get_case_dir(7, "resilient", 8, base_run_dir=base_run_dir)

    assert raw_list.exists()
    assert pointlike.exists()
    assert (case_dir / "clumpy_params.template.txt").exists()
    assert not (case_dir / "preparation_manifest.json").exists()
    assert not os.path.lexists(case_dir / ".preparation.lock")


def test_prepare_case_refuses_all_existing_targets_without_changes(tmp_path):
    input_h5, template = make_case_inputs(tmp_path)
    base_run_dir = tmp_path / "outputs" / "clumpy"

    manifest_path, manifest = prep.prepare_case(
        repop_id=7,
        scenario="resilient",
        nside=8,
        base_h5_dir=tmp_path / "outputs",
        base_run_dir=base_run_dir,
        template_path=template,
        repository_root=tmp_path,
        chunk_size=2,
    )

    manifest_base = manifest_path.parent / manifest["paths"]["base"]
    protected_paths = [
        (manifest_base / artifact["path"]).resolve()
        for artifact in manifest["artifacts"].values()
    ] + [manifest_path]
    checksums_before = {path: checksum(path) for path in protected_paths}

    with pytest.raises(FileExistsError) as exc_info:
        prep.prepare_case(
            repop_id=7,
            scenario="resilient",
            nside=8,
            base_h5_dir=tmp_path / "outputs",
            base_run_dir=base_run_dir,
            template_path=template,
            repository_root=tmp_path,
            chunk_size=2,
        )

    message = str(exc_info.value)
    assert "--force is not implemented yet" in message
    for path in protected_paths:
        assert str(path) in message
        assert checksum(path) == checksums_before[path]


def test_prepare_wrapper_runs_without_clumpy(tmp_path):
    repository_root = tmp_path / "repository"
    make_case_inputs(repository_root)
    scripts_dir = repository_root / "scripts"
    scripts_dir.mkdir()

    for name in ("prepare_clumpy_case.sh", "prepare_subhalo_components.py"):
        shutil.copy2(REPOSITORY_ROOT / "scripts" / name, scripts_dir / name)

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    environment["PATH"] = "/usr/bin:/bin"
    python_marker = tmp_path / "python-used"
    python_wrapper = tmp_path / "explicit-python"
    python_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"touch {shlex.quote(str(python_marker))}\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    environment["PYTHON_EXECUTABLE"] = str(python_wrapper)

    external_cwd = tmp_path / "external-working-directory"
    external_cwd.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(scripts_dir / "prepare_clumpy_case.sh"),
            "7",
            "resilient",
            "--nside",
            "8",
        ],
        cwd=external_cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = (
        repository_root
        / "outputs"
        / "clumpy"
        / "resilient"
        / "cases"
        / "repop_0007_nside8"
        / "preparation_manifest.json"
    )
    assert manifest.exists()
    manifest_payload = strict_json_loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["scientific_configuration"]["cuts"][
        "extended_cut_f"
    ] == pytest.approx(1e-3)
    assert manifest_payload["scientific_configuration"]["cuts"][
        "pointlike_cut_f"
    ] == pytest.approx(1e-3)
    assert python_marker.exists()
    assert "CLUMPY was not executed." in result.stdout
    assert not (
        repository_root / "outputs" / "clumpy" / "resilient" / "params"
    ).exists()
