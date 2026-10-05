import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WRAPPER = REPOSITORY_ROOT / "scripts" / "run_clumpy_one_case.sh"


FAKE_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail

script_name="$(basename -- "$1")"
shift

repop_id="$1"
scenario="$2"
printf -v repop_tag 'repop_%04d' "$repop_id"
nside="${NSIDE}"
case_dir="${FAKE_REPO_ROOT}/outputs/clumpy/${scenario}"

case "$script_name" in
    prepare_subhalo_components.py)
        mkdir -p "${case_dir}/lists/raw" "${case_dir}/pointlike"
        : > "${case_dir}/lists/raw/${repop_tag}_raw_nopointlike_nside${nside}.txt"
        printf 'fake fits\n' > "${case_dir}/pointlike/${repop_tag}_pointlike_nside${nside}.fits"
        ;;
    generate_clumpy_params.py)
        stage="$3"
        mkdir -p "${case_dir}/params/generated"
        : > "${case_dir}/params/generated/${repop_tag}_${stage}_params_nside${nside}.txt"
        ;;
    correct_rhos_from_clumpy_raw.py)
        mkdir -p "${case_dir}/lists/corrected"
        : > "${case_dir}/lists/corrected/${repop_tag}_rhocorr_nside${nside}.txt"
        ;;
    combine_clumpy_pointlike.py)
        total_dir="${case_dir}/outputs/total/${repop_tag}_nside${nside}"
        mkdir -p "$total_dir"
        printf 'fake fits\n' > "${total_dir}/auriga_total_nside${nside}.fits"
        ;;
    *)
        echo "Unexpected Python entry point: $script_name" >&2
        exit 2
        ;;
esac
"""


FAKE_CLUMPY = r"""#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Expected exactly three CLUMPY arguments, received $#" >&2
    exit 2
fi

printf '%s\t%s\t%s\t%s\n' "$CLUMPY_DATA" "$1" "$2" "$3" >> "$FAKE_CALL_LOG"

param_file="$3"
repop_tag="repop_0007"
nside="${NSIDE}"
case_dir="${FAKE_REPO_ROOT}/outputs/clumpy/resilient"
basename="annihil_gal2D_LOS0_0_FOV360x180_nside${nside}"

case "$param_file" in
    *_raw_params_nside${nside}.txt)
        output_dir="${case_dir}/outputs/raw_clumpy/${repop_tag}_nside${nside}"
        mkdir -p "$output_dir"
        : > "${output_dir}/${basename}.halo_rendered.log"
        ;;
    *_corrected_params_nside${nside}.txt)
        output_dir="${case_dir}/outputs/corrected_clumpy/${repop_tag}_nside${nside}"
        mkdir -p "$output_dir"
        : > "${output_dir}/${basename}.halo_rendered.log"
        printf 'fake fits\n' > "${output_dir}/${basename}.fits"
        ;;
    *)
        echo "Unexpected parameter file: $param_file" >&2
        exit 2
        ;;
esac
"""


def write_executable(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def make_test_repository(tmp_path):
    repository = tmp_path / "auriga-pipeline"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(SOURCE_WRAPPER, scripts / SOURCE_WRAPPER.name)

    fake_python = tmp_path / "fake-python"
    write_executable(fake_python, FAKE_PYTHON)

    call_log = tmp_path / "clumpy-calls.tsv"
    return repository, fake_python, call_log


def make_clumpy_tree(tmp_path, executable_relative_path, *, with_data=True):
    clumpy_root = tmp_path / "CLUMPY"
    executable = clumpy_root / executable_relative_path
    write_executable(executable, FAKE_CLUMPY)

    data_dir = clumpy_root / "data"
    if with_data:
        data_dir.mkdir(parents=True)
        (data_dir / "list_generic.txt").write_text("test\n", encoding="utf-8")
        (data_dir / "healpix").mkdir()

    return executable, data_dir


def run_wrapper(repository, fake_python, call_log, clumpy_executable, **env_updates):
    environment = os.environ.copy()
    environment.pop("CLUMPY_DATA", None)
    environment.update(
        {
            "CLUMPY_EXECUTABLE": str(clumpy_executable),
            "PYTHON_EXECUTABLE": str(fake_python),
            "FAKE_REPO_ROOT": str(repository),
            "FAKE_CALL_LOG": str(call_log),
            "NSIDE": "8",
        }
    )
    environment.update(env_updates)

    return subprocess.run(
        ["bash", str(repository / "scripts" / SOURCE_WRAPPER.name), "7", "resilient"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def assert_two_clumpy_stages(result, call_log, expected_data):
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [line.split("\t") for line in call_log.read_text().splitlines()]
    assert len(calls) == 2

    raw, corrected = calls
    assert raw[0] == str(expected_data.resolve())
    assert corrected[0] == str(expected_data.resolve())
    assert raw[1:3] == ["-g6", "-i"]
    assert corrected[1:3] == ["-g6", "-i"]
    assert raw[3].endswith("repop_0007_raw_params_nside8.txt")
    assert corrected[3].endswith("repop_0007_corrected_params_nside8.txt")


def case_paths(repository):
    case_dir = repository / "outputs" / "clumpy" / "resilient"
    repop_dir = "repop_0007_nside8"
    basename = "annihil_gal2D_LOS0_0_FOV360x180_nside8"

    return {
        "raw_list": (
            case_dir / "lists" / "raw" / "repop_0007_raw_nopointlike_nside8.txt"
        ),
        "corrected_fits": (
            case_dir / "outputs" / "corrected_clumpy" / repop_dir / f"{basename}.fits"
        ),
        "pointlike_fits": (
            case_dir / "pointlike" / "repop_0007_pointlike_nside8.fits"
        ),
        "total_fits": (
            case_dir / "outputs" / "total" / repop_dir / "auriga_total_nside8.fits"
        ),
    }


def test_existing_corrected_and_pointlike_fits_resume_final_combination(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = case_paths(repository)

    paths["corrected_fits"].parent.mkdir(parents=True)
    paths["corrected_fits"].write_text("fake corrected fits\n", encoding="utf-8")
    paths["pointlike_fits"].parent.mkdir(parents=True)
    paths["pointlike_fits"].write_text("fake pointlike fits\n", encoding="utf-8")

    result = run_wrapper(repository, fake_python, call_log, executable)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "Resuming from existing corrected CLUMPY and pointlike FITS files."
        in result.stdout
    )
    assert paths["total_fits"].read_text(encoding="utf-8") == "fake fits\n"
    assert not paths["raw_list"].exists()
    assert not call_log.exists()


def test_existing_total_fits_without_corrected_fits_refuses_all_stages(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = case_paths(repository)

    paths["total_fits"].parent.mkdir(parents=True)
    paths["total_fits"].write_text("completed total fits\n", encoding="utf-8")

    result = run_wrapper(repository, fake_python, call_log, executable)

    assert result.returncode != 0
    assert "final total FITS already exists and is non-empty" in result.stdout
    assert paths["total_fits"].read_text(encoding="utf-8") == "completed total fits\n"
    assert not paths["corrected_fits"].exists()
    assert not paths["raw_list"].exists()
    assert not call_log.exists()


@pytest.mark.parametrize("pointlike_state", ["missing", "empty"])
def test_resume_refuses_missing_or_empty_pointlike_fits(tmp_path, pointlike_state):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = case_paths(repository)

    paths["corrected_fits"].parent.mkdir(parents=True)
    paths["corrected_fits"].write_text("fake corrected fits\n", encoding="utf-8")

    if pointlike_state == "empty":
        paths["pointlike_fits"].parent.mkdir(parents=True)
        paths["pointlike_fits"].touch()

    result = run_wrapper(repository, fake_python, call_log, executable)

    assert result.returncode != 0
    assert "pointlike FITS is missing or empty" in result.stdout
    assert not paths["total_fits"].exists()
    assert not paths["raw_list"].exists()
    assert not call_log.exists()


def test_explicit_valid_clumpy_data_is_respected(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable = tmp_path / "custom-install" / "custom-clumpy"
    write_executable(executable, FAKE_CLUMPY)

    explicit_data = tmp_path / "explicit-data"
    explicit_data.mkdir()
    (explicit_data / "list_generic.txt").write_text("test\n", encoding="utf-8")
    (explicit_data / "healpix").mkdir()

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        CLUMPY_DATA=str(explicit_data),
    )

    assert_two_clumpy_stages(result, call_log, explicit_data)
    assert f"CLUMPY_DATA: {explicit_data.resolve()}" in result.stdout


@pytest.mark.parametrize("executable_path", ["bin/clumpy", "build/cli/clumpy"])
def test_clumpy_data_is_inferred_from_source_tree_layout(tmp_path, executable_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, data_dir = make_clumpy_tree(tmp_path, executable_path)

    result = run_wrapper(repository, fake_python, call_log, executable)

    assert_two_clumpy_stages(result, call_log, data_dir)
    assert f"CLUMPY:    {executable.resolve()}" in result.stdout
    assert f"CLUMPY_DATA: {data_dir.resolve()}" in result.stdout


@pytest.mark.parametrize("explicit", [False, True])
def test_invalid_or_missing_data_fails_before_clumpy_execution(tmp_path, explicit):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, data_dir = make_clumpy_tree(
        tmp_path,
        "bin/clumpy",
        with_data=False,
    )

    updates = {"CLUMPY_DATA": str(data_dir)} if explicit else {}
    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        **updates,
    )

    assert result.returncode != 0
    assert "CLUMPY_DATA is not a valid CLUMPY data directory" in result.stderr
    assert not call_log.exists()
    assert not (repository / "outputs").exists()
