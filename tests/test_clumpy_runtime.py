import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WRAPPER = REPOSITORY_ROOT / "scripts" / "run_clumpy_one_case.sh"
SOURCE_BATCH_WRAPPER = REPOSITORY_ROOT / "scripts" / "run_clumpy_batch.sh"


FAKE_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail

script_name="$(basename -- "$1")"
shift
printf '%s\n' "$script_name" >> "$FAKE_PYTHON_CALL_LOG"
printf '%s\t%s\n' "$script_name" "$*" >> "$FAKE_PYTHON_ARG_LOG"

repop_id="$1"
scenario="$2"
all_args=("$@")
printf -v repop_tag 'repop_%04d' "$repop_id"
nside="${NSIDE}"
case_dir="${FAKE_REPO_ROOT}/outputs/clumpy/${scenario}"

option_value() {
    local option="$1"
    local index
    for ((index=0; index<${#all_args[@]}; index++)); do
        if [ "${all_args[index]}" = "$option" ]; then
            printf '%s\n' "${all_args[index+1]}"
            return 0
        fi
    done
    return 1
}

has_option() {
    local option="$1"
    local argument
    for argument in "${all_args[@]}"; do
        if [ "$argument" = "$option" ]; then
            return 0
        fi
    done
    return 1
}

case "$script_name" in
    prepare_subhalo_components.py)
        if has_option --validate-pointlike-only; then
            test -s "${case_dir}/pointlike/${repop_tag}_pointlike_nside${nside}.fits"
            exit 0
        fi
        mkdir -p "${case_dir}/lists/raw" "${case_dir}/pointlike"
        printf 'raw list\n' > "${case_dir}/lists/raw/${repop_tag}_raw_nopointlike_nside${nside}.txt"
        printf 'fake fits\n' > "${case_dir}/pointlike/${repop_tag}_pointlike_nside${nside}.fits"
        ;;
    generate_clumpy_params.py)
        stage="$3"
        output_dir="$(option_value --output-dir || true)"
        output_param="$(option_value --output-param || true)"
        if [ -z "$output_dir" ]; then
            output_dir="${case_dir}/outputs/${stage}_clumpy/${repop_tag}_nside${nside}"
        fi
        if [ -z "$output_param" ]; then
            output_param="${case_dir}/params/generated/${repop_tag}_${stage}_params_nside${nside}.txt"
        fi
        mkdir -p "$(dirname "$output_param")"
        if ! has_option --defer-output-dir-creation; then
            mkdir -p "$output_dir"
        fi
        printf 'OUTPUT_DIR=%s\n' "$output_dir" > "$output_param"
        ;;
    correct_rhos_from_clumpy_raw.py)
        if has_option --validate-only; then
            if [ "${FAKE_CORRECTION_VALIDATE_FAIL:-0}" = "1" ]; then
                echo "synthetic raw/HDF5 validation failure" >&2
                exit 17
            fi
            exit 0
        fi
        output_list="$(option_value --output-list || true)"
        output_dir="$(option_value --output-dir || true)"
        if [ -z "$output_list" ]; then
            output_list="${case_dir}/lists/corrected/${repop_tag}_rhocorr_nside${nside}.txt"
        fi
        if [ -z "$output_dir" ]; then
            output_dir="${case_dir}/logs/correction/${repop_tag}_nside${nside}"
        fi
        mkdir -p "$(dirname "$output_list")" "$output_dir"
        printf 'corrected list\n' > "$output_list"
        printf 'diagnostics\n' > "$output_dir/matched.csv"
        ;;
    combine_clumpy_pointlike.py)
        if [ "${FAKE_COMBINE_FAIL:-0}" = "1" ]; then
            echo "synthetic combination failure" >&2
            exit 23
        fi
        output_fits="$(option_value --output-fits || true)"
        if [ -z "$output_fits" ]; then
            output_fits="${case_dir}/outputs/total/${repop_tag}_nside${nside}/auriga_total_nside${nside}.fits"
        fi
        mkdir -p "$(dirname "$output_fits")"
        if [ "${FAKE_COMBINE_EMPTY:-0}" = "1" ]; then
            : > "$output_fits"
            exit 0
        fi
        printf 'fake fits\n' > "$output_fits"
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
nside="${NSIDE}"
basename="annihil_gal2D_LOS0_0_FOV360x180_nside${nside}"
output_dir="$(sed -n 's/^OUTPUT_DIR=//p' "$param_file")"

case "$param_file" in
    *_raw_params_nside${nside}.txt)
        mkdir -p "$output_dir"
        printf '0 hydro_res_00001 DSPH 1 1\n' > "${output_dir}/${basename}.halo_rendered.log"
        printf 'raw sidecar\n' > "${output_dir}/${basename}.list"
        if [ -n "${FAKE_RAW_FITS_SYMLINK_TARGET:-}" ]; then
            ln -s "$FAKE_RAW_FITS_SYMLINK_TARGET" "${output_dir}/${basename}.fits"
        else
            printf 'fake raw fits\n' > "${output_dir}/${basename}.fits"
        fi
        ;;
    *_corrected_params_nside${nside}.txt)
        if [ "${FAKE_CLUMPY_FAIL_CORRECTED:-0}" = "1" ]; then
            echo "synthetic corrected failure" >&2
            exit 9
        fi
        mkdir -p "$output_dir"
        printf '0 hydro_res_00001 DSPH 1 1\n' > "${output_dir}/${basename}.halo_rendered.log"
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


def run_wrapper(
    repository,
    fake_python,
    call_log,
    clumpy_executable,
    *,
    from_correction=False,
    discard_stale_downstream=False,
    cleanup_raw_fits=False,
    **env_updates,
):
    environment = os.environ.copy()
    environment.pop("CLUMPY_DATA", None)
    environment.update(
        {
            "CLUMPY_EXECUTABLE": str(clumpy_executable),
            "PYTHON_EXECUTABLE": str(fake_python),
            "FAKE_REPO_ROOT": str(repository),
            "FAKE_CALL_LOG": str(call_log),
            "FAKE_PYTHON_CALL_LOG": str(call_log.with_suffix(".python.tsv")),
            "FAKE_PYTHON_ARG_LOG": str(call_log.with_suffix(".python-args.tsv")),
            "NSIDE": "8",
        }
    )
    environment.update(env_updates)

    command = [
        "bash",
        str(repository / "scripts" / SOURCE_WRAPPER.name),
        "7",
        "resilient",
    ]
    if from_correction:
        command.append("--from-correction")
    if discard_stale_downstream:
        command.append("--discard-stale-downstream")
    if cleanup_raw_fits:
        command.append("--cleanup-raw-fits")

    return subprocess.run(
        command,
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
        "raw_log": (
            case_dir
            / "outputs"
            / "raw_clumpy"
            / repop_dir
            / f"{basename}.halo_rendered.log"
        ),
        "raw_fits": (
            case_dir
            / "outputs"
            / "raw_clumpy"
            / repop_dir
            / f"{basename}.fits"
        ),
        "raw_sidecar": (
            case_dir
            / "outputs"
            / "raw_clumpy"
            / repop_dir
            / f"{basename}.list"
        ),
        "raw_output_dir": case_dir / "outputs" / "raw_clumpy" / repop_dir,
        "raw_param": (
            case_dir / "params" / "generated" / "repop_0007_raw_params_nside8.txt"
        ),
        "raw_stdout": (
            case_dir / "logs" / "raw_clumpy" / "repop_0007_raw_clumpy_nside8.log"
        ),
        "corrected_list": (
            case_dir / "lists" / "corrected" / "repop_0007_rhocorr_nside8.txt"
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
        "corrected_log": (
            case_dir
            / "outputs"
            / "corrected_clumpy"
            / repop_dir
            / f"{basename}.halo_rendered.log"
        ),
        "corrected_output_dir": (
            case_dir / "outputs" / "corrected_clumpy" / repop_dir
        ),
        "corrected_param": (
            case_dir
            / "params"
            / "generated"
            / "repop_0007_corrected_params_nside8.txt"
        ),
        "corrected_stdout": (
            case_dir
            / "logs"
            / "corrected_clumpy"
            / "repop_0007_corrected_clumpy_nside8.log"
        ),
        "correction_dir": case_dir / "logs" / "correction" / repop_dir,
        "total_output_dir": case_dir / "outputs" / "total" / repop_dir,
    }


def create_downstream_prerequisites(repository):
    paths = case_paths(repository)
    catalog = repository / "outputs" / "repop_0007" / "fullrepop_hydro_resilient.h5"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("catalogue\n", encoding="utf-8")
    paths["raw_list"].parent.mkdir(parents=True)
    paths["raw_list"].write_text("original raw list\n", encoding="utf-8")
    paths["raw_log"].parent.mkdir(parents=True)
    paths["raw_log"].write_text("original raw rendered log\n", encoding="utf-8")
    paths["pointlike_fits"].parent.mkdir(parents=True)
    paths["pointlike_fits"].write_text("original pointlike fits\n", encoding="utf-8")
    return paths


def create_stale_downstream_products(paths):
    paths["corrected_list"].parent.mkdir(parents=True, exist_ok=True)
    paths["corrected_list"].write_text("stale corrected list\n", encoding="utf-8")
    paths["corrected_param"].parent.mkdir(parents=True, exist_ok=True)
    paths["corrected_param"].write_text("stale corrected params\n", encoding="utf-8")
    paths["corrected_fits"].parent.mkdir(parents=True, exist_ok=True)
    paths["corrected_fits"].write_text("stale corrected fits\n", encoding="utf-8")
    paths["corrected_log"].write_text("stale corrected log\n", encoding="utf-8")
    paths["corrected_stdout"].parent.mkdir(parents=True, exist_ok=True)
    paths["corrected_stdout"].write_text("stale corrected stdout\n", encoding="utf-8")
    paths["correction_dir"].mkdir(parents=True, exist_ok=True)
    (paths["correction_dir"] / "old-diagnostics.csv").write_text(
        "stale diagnostics\n", encoding="utf-8"
    )
    paths["total_fits"].parent.mkdir(parents=True, exist_ok=True)
    paths["total_fits"].write_text("stale total fits\n", encoding="utf-8")


def create_optional_raw_products(paths):
    paths["raw_param"].parent.mkdir(parents=True, exist_ok=True)
    paths["raw_param"].write_text("original raw params\n", encoding="utf-8")
    paths["raw_stdout"].parent.mkdir(parents=True, exist_ok=True)
    paths["raw_stdout"].write_text("original raw stdout\n", encoding="utf-8")
    (paths["raw_output_dir"] / "raw-map.fits").write_text(
        "original raw map\n", encoding="utf-8"
    )


def test_default_run_preserves_primary_raw_fits(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")

    result = run_wrapper(repository, fake_python, call_log, executable)
    paths = case_paths(repository)

    assert result.returncode == 0, result.stdout + result.stderr
    assert paths["raw_fits"].read_text() == "fake raw fits\n"
    assert paths["raw_log"].is_file()
    assert paths["total_fits"].is_file()


def test_cleanup_removes_only_primary_raw_fits_after_success(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = case_paths(repository)
    unrelated = (
        repository
        / "outputs/clumpy/fragile/outputs/raw_clumpy/repop_0008_nside8"
        / paths["raw_fits"].name
    )
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("unrelated raw fits\n", encoding="utf-8")

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        cleanup_raw_fits=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Removed primary raw CLUMPY FITS" in result.stdout
    assert not paths["raw_fits"].exists()
    for key in (
        "raw_log",
        "raw_sidecar",
        "raw_stdout",
        "raw_param",
        "raw_list",
        "corrected_list",
        "corrected_fits",
        "corrected_log",
        "corrected_stdout",
        "pointlike_fits",
        "total_fits",
    ):
        assert paths[key].is_file(), key
    assert paths["correction_dir"].is_dir()
    assert unrelated.read_text() == "unrelated raw fits\n"


@pytest.mark.parametrize(
    ("failure_env", "expected_error"),
    [
        ({"FAKE_CLUMPY_FAIL_CORRECTED": "1"}, "synthetic corrected failure"),
        ({"FAKE_COMBINE_FAIL": "1"}, "synthetic combination failure"),
        (
            {"FAKE_COMBINE_EMPTY": "1"},
            "total FITS was not created or is empty",
        ),
    ],
)
def test_cleanup_never_runs_after_downstream_failure(
    tmp_path, failure_env, expected_error
):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        cleanup_raw_fits=True,
        **failure_env,
    )
    paths = case_paths(repository)

    assert result.returncode != 0
    assert expected_error in result.stdout + result.stderr
    assert paths["raw_fits"].read_text() == "fake raw fits\n"


def test_cleanup_allows_symlinked_clumpy_output_root(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    (repository / "outputs").mkdir()
    external_clumpy = tmp_path / "external-output" / "clumpy"
    external_clumpy.mkdir(parents=True)
    (repository / "outputs" / "clumpy").symlink_to(
        external_clumpy, target_is_directory=True
    )

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        cleanup_raw_fits=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not case_paths(repository)["raw_fits"].exists()
    assert case_paths(repository)["raw_log"].is_file()


def test_cleanup_rejects_raw_fits_symlink_without_touching_target(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    outside = tmp_path / "must-survive.fits"
    outside.write_text("outside raw map\n", encoding="utf-8")

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        cleanup_raw_fits=True,
        FAKE_RAW_FITS_SYMLINK_TARGET=str(outside),
    )
    paths = case_paths(repository)

    assert result.returncode != 0
    assert "refusing to remove a symlink" in result.stderr
    assert paths["raw_fits"].is_symlink()
    assert outside.read_text() == "outside raw map\n"
    assert paths["total_fits"].is_file()


def test_cleanup_rejects_raw_directory_redirected_to_another_repopulation(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = create_downstream_prerequisites(repository)
    paths["raw_fits"].write_text("other repop raw map\n", encoding="utf-8")
    other_repop_dir = paths["raw_output_dir"].with_name("repop_9999_nside8")
    paths["raw_output_dir"].rename(other_repop_dir)
    paths["raw_output_dir"].symlink_to(other_repop_dir, target_is_directory=True)

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        from_correction=True,
        cleanup_raw_fits=True,
    )

    assert result.returncode != 0
    assert "unexpected resolved directory" in result.stderr
    other_raw_fits = other_repop_dir / paths["raw_fits"].name
    assert other_raw_fits.read_text() == "other repop raw map\n"


def test_from_correction_is_idempotent_after_raw_fits_cleanup(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    catalog = repository / "outputs/repop_0007/fullrepop_hydro_resilient.h5"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("catalogue\n", encoding="utf-8")

    first = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        cleanup_raw_fits=True,
    )
    paths = case_paths(repository)
    assert first.returncode == 0, first.stdout + first.stderr
    assert not paths["raw_fits"].exists()

    second = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        from_correction=True,
        cleanup_raw_fits=True,
    )

    assert second.returncode == 0, second.stdout + second.stderr
    assert "already absent; no cleanup needed" in second.stdout
    assert not paths["raw_fits"].exists()
    assert paths["raw_log"].is_file()
    assert paths["corrected_fits"].is_file()
    assert paths["total_fits"].is_file()


def test_from_correction_reuses_raw_and_replaces_stale_downstream_only(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = create_downstream_prerequisites(repository)

    paths["corrected_list"].parent.mkdir(parents=True)
    paths["corrected_list"].write_text("stale corrected list\n", encoding="utf-8")
    paths["corrected_fits"].parent.mkdir(parents=True)
    paths["corrected_fits"].write_text("stale corrected fits\n", encoding="utf-8")
    paths["corrected_log"].write_text("stale corrected log\n", encoding="utf-8")
    paths["total_fits"].parent.mkdir(parents=True)
    paths["total_fits"].write_text("stale total fits\n", encoding="utf-8")

    raw_list_before = paths["raw_list"].read_bytes()
    raw_log_before = paths["raw_log"].read_bytes()
    pointlike_before = paths["pointlike_fits"].read_bytes()

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        from_correction=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Downstream-only correction rerun completed successfully" in result.stdout
    assert paths["raw_list"].read_bytes() == raw_list_before
    assert paths["raw_log"].read_bytes() == raw_log_before
    assert paths["pointlike_fits"].read_bytes() == pointlike_before
    assert paths["corrected_list"].read_text() == "corrected list\n"
    assert paths["corrected_fits"].read_text() == "fake fits\n"
    assert paths["total_fits"].read_text() == "fake fits\n"

    clumpy_calls = [line.split("\t") for line in call_log.read_text().splitlines()]
    assert len(clumpy_calls) == 1
    assert clumpy_calls[0][3].endswith("repop_0007_corrected_params_nside8.txt")
    python_calls = call_log.with_suffix(".python.tsv").read_text().splitlines()
    assert python_calls.count("prepare_subhalo_components.py") == 1
    assert python_calls.count("correct_rhos_from_clumpy_raw.py") == 2
    assert python_calls.count("generate_clumpy_params.py") >= 1
    assert python_calls.count("combine_clumpy_pointlike.py") == 1
    python_arg_calls = call_log.with_suffix(".python-args.tsv").read_text().splitlines()
    prepare_calls = [
        line for line in python_arg_calls
        if line.startswith("prepare_subhalo_components.py\t")
    ]
    assert len(prepare_calls) == 1
    assert "--validate-pointlike-only" in prepare_calls[0]

    backups = list(
        (repository / "outputs" / "clumpy" / "resilient" / "backups" / "from_correction").glob("*")
    )
    assert len(backups) == 1
    assert (backups[0] / "corrected_list.txt").read_text() == "stale corrected list\n"
    assert (
        backups[0] / "corrected_clumpy_output" / paths["corrected_fits"].name
    ).read_text() == "stale corrected fits\n"
    assert (
        backups[0] / "total_output" / paths["total_fits"].name
    ).read_text() == "stale total fits\n"


def test_from_correction_missing_raw_prerequisite_aborts_without_replacement(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = create_downstream_prerequisites(repository)
    paths["raw_log"].unlink()
    paths["total_fits"].parent.mkdir(parents=True)
    paths["total_fits"].write_text("stale total fits\n", encoding="utf-8")

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        from_correction=True,
    )

    assert result.returncode != 0
    assert "required downstream-rerun input is missing or empty" in result.stderr
    assert paths["total_fits"].read_text() == "stale total fits\n"
    assert not call_log.exists()
    assert not call_log.with_suffix(".python.tsv").exists()


def test_from_correction_failure_leaves_stable_downstream_untouched(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = create_downstream_prerequisites(repository)
    paths["raw_fits"].write_text("original raw fits\n", encoding="utf-8")
    paths["corrected_fits"].parent.mkdir(parents=True)
    paths["corrected_fits"].write_text("stale corrected fits\n", encoding="utf-8")
    paths["total_fits"].parent.mkdir(parents=True)
    paths["total_fits"].write_text("stale total fits\n", encoding="utf-8")

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        from_correction=True,
        cleanup_raw_fits=True,
        FAKE_CLUMPY_FAIL_CORRECTED="1",
    )

    assert result.returncode != 0
    assert paths["corrected_fits"].read_text() == "stale corrected fits\n"
    assert paths["total_fits"].read_text() == "stale total fits\n"
    assert paths["raw_list"].read_text() == "original raw list\n"
    assert paths["raw_log"].read_text() == "original raw rendered log\n"
    assert paths["raw_fits"].read_text() == "original raw fits\n"


def test_discard_stale_downstream_removes_only_exact_downstream_paths(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = create_downstream_prerequisites(repository)
    create_optional_raw_products(paths)
    paths["raw_fits"].write_text("primary raw fits\n", encoding="utf-8")
    create_stale_downstream_products(paths)

    unrelated = (
        repository
        / "outputs/clumpy/resilient/outputs/total/repop_9999_nside8"
        / "auriga_total_nside8.fits"
    )
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("unrelated total\n", encoding="utf-8")

    protected = {
        key: path.read_bytes()
        for key, path in paths.items()
        if key in {
            "raw_list",
            "raw_log",
            "raw_param",
            "raw_stdout",
            "pointlike_fits",
        }
    }
    raw_map = paths["raw_output_dir"] / "raw-map.fits"
    raw_map_before = raw_map.read_bytes()

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        from_correction=True,
        discard_stale_downstream=True,
        cleanup_raw_fits=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WILL PRESERVE:" in result.stdout
    assert "WILL REMOVE:" in result.stdout
    assert "no backup was retained" in result.stdout
    for key, contents in protected.items():
        assert paths[key].read_bytes() == contents
    assert raw_map.read_bytes() == raw_map_before
    assert not paths["raw_fits"].exists()
    assert paths["raw_log"].read_bytes() == protected["raw_log"]
    assert unrelated.read_text(encoding="utf-8") == "unrelated total\n"
    assert paths["corrected_list"].read_text() == "corrected list\n"
    assert paths["corrected_fits"].read_text() == "fake fits\n"
    assert paths["total_fits"].read_text() == "fake fits\n"
    assert paths["corrected_stdout"].is_file()
    assert not (
        repository / "outputs/clumpy/resilient/backups/from_correction"
    ).exists()
    assert not list(
        (repository / "outputs/clumpy/resilient").glob(
            ".from_correction_repop_0007_nside8.*"
        )
    )

    clumpy_calls = call_log.read_text().splitlines()
    assert len(clumpy_calls) == 1
    assert "corrected_params_nside8.txt" in clumpy_calls[0]
    python_arg_calls = call_log.with_suffix(".python-args.tsv").read_text().splitlines()
    prepare_calls = [
        line for line in python_arg_calls
        if line.startswith("prepare_subhalo_components.py\t")
    ]
    assert len(prepare_calls) == 1
    assert "--validate-pointlike-only" in prepare_calls[0]


def test_discard_validation_failure_precedes_any_deletion(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = create_downstream_prerequisites(repository)
    create_stale_downstream_products(paths)

    stale_before = {
        "corrected_list": paths["corrected_list"].read_bytes(),
        "corrected_fits": paths["corrected_fits"].read_bytes(),
        "total_fits": paths["total_fits"].read_bytes(),
    }
    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        from_correction=True,
        discard_stale_downstream=True,
        FAKE_CORRECTION_VALIDATE_FAIL="1",
    )

    assert result.returncode != 0
    assert "synthetic raw/HDF5 validation failure" in result.stderr
    for key, contents in stale_before.items():
        assert paths[key].read_bytes() == contents
    assert not call_log.exists()


def test_discard_rejects_resolved_path_outside_case_tree(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")
    paths = create_downstream_prerequisites(repository)
    paths["corrected_list"].parent.mkdir(parents=True)
    paths["corrected_list"].write_text("stale corrected list\n", encoding="utf-8")

    outside = tmp_path / "outside-correction"
    outside.mkdir()
    outside_sentinel = outside / "must-survive.csv"
    outside_sentinel.write_text("outside\n", encoding="utf-8")
    paths["correction_dir"].parent.mkdir(parents=True)
    paths["correction_dir"].symlink_to(outside, target_is_directory=True)

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        from_correction=True,
        discard_stale_downstream=True,
    )

    assert result.returncode != 0
    assert "outside case tree" in result.stderr
    assert outside_sentinel.read_text(encoding="utf-8") == "outside\n"
    assert paths["corrected_list"].read_text() == "stale corrected list\n"
    assert not call_log.exists()


def test_discard_stale_downstream_requires_from_correction(tmp_path):
    repository, fake_python, call_log = make_test_repository(tmp_path)
    executable, _ = make_clumpy_tree(tmp_path, "bin/clumpy")

    result = run_wrapper(
        repository,
        fake_python,
        call_log,
        executable,
        discard_stale_downstream=True,
    )

    assert result.returncode == 2
    assert "requires --from-correction" in result.stderr
    assert not (repository / "outputs").exists()


def test_batch_from_correction_does_not_skip_existing_total(tmp_path):
    repository = tmp_path / "auriga-pipeline"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(SOURCE_BATCH_WRAPPER, scripts / SOURCE_BATCH_WRAPPER.name)
    invocation_log = tmp_path / "one-case-invocation.txt"
    write_executable(
        scripts / SOURCE_WRAPPER.name,
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" > \"$BATCH_ONE_CASE_LOG\"\n",
    )

    catalog_root = tmp_path / "catalogues"
    catalog = catalog_root / "repop_0007" / "fullrepop_hydro_resilient.h5"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("catalogue\n", encoding="utf-8")
    total = (
        repository
        / "outputs/clumpy/resilient/outputs/total/repop_0007_nside8"
        / "auriga_total_nside8.fits"
    )
    total.parent.mkdir(parents=True)
    total.write_text("stale total\n", encoding="utf-8")

    environment = os.environ.copy()
    environment.update(
        {
            "BATCH_ONE_CASE_LOG": str(invocation_log),
            "CATALOG_ROOT": str(catalog_root),
            "NSIDE": "8",
            "LAUNCH_DELAY_SECONDS": "0",
            "POLL_SECONDS": "0",
            "SKIP_COMPLETED": "1",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(scripts / SOURCE_BATCH_WRAPPER.name),
            "7",
            "7",
            "resilient",
            "1",
            "--from-correction",
        ],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert invocation_log.read_text().strip() == "7 resilient --from-correction"
    assert "Skipped:    0" in result.stdout


def test_batch_cleanup_option_does_not_change_skip_completed(tmp_path):
    repository = tmp_path / "auriga-pipeline"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(SOURCE_BATCH_WRAPPER, scripts / SOURCE_BATCH_WRAPPER.name)
    invocation_log = tmp_path / "one-case-invocation.txt"
    write_executable(
        scripts / SOURCE_WRAPPER.name,
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" > \"$BATCH_ONE_CASE_LOG\"\n",
    )
    catalog_root = tmp_path / "catalogues"
    catalog = catalog_root / "repop_0007" / "fullrepop_hydro_resilient.h5"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("catalogue\n", encoding="utf-8")
    total = (
        repository
        / "outputs/clumpy/resilient/outputs/total/repop_0007_nside8"
        / "auriga_total_nside8.fits"
    )
    total.parent.mkdir(parents=True)
    total.write_text("completed total\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "BATCH_ONE_CASE_LOG": str(invocation_log),
            "CATALOG_ROOT": str(catalog_root),
            "NSIDE": "8",
            "LAUNCH_DELAY_SECONDS": "0",
            "POLL_SECONDS": "0",
            "SKIP_COMPLETED": "1",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(scripts / SOURCE_BATCH_WRAPPER.name),
            "7",
            "7",
            "resilient",
            "1",
            "--cleanup-raw-fits",
        ],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not invocation_log.exists()
    assert "Skipped:    1" in result.stdout


def test_batch_propagates_discard_and_cleanup_options(tmp_path):
    repository = tmp_path / "auriga-pipeline"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(SOURCE_BATCH_WRAPPER, scripts / SOURCE_BATCH_WRAPPER.name)
    invocation_log = tmp_path / "one-case-invocation.txt"
    write_executable(
        scripts / SOURCE_WRAPPER.name,
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" > \"$BATCH_ONE_CASE_LOG\"\n",
    )

    catalog_root = tmp_path / "catalogues"
    catalog = catalog_root / "repop_0007" / "fullrepop_hydro_resilient.h5"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("catalogue\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        {
            "BATCH_ONE_CASE_LOG": str(invocation_log),
            "CATALOG_ROOT": str(catalog_root),
            "NSIDE": "8",
            "LAUNCH_DELAY_SECONDS": "0",
            "POLL_SECONDS": "0",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(scripts / SOURCE_BATCH_WRAPPER.name),
            "7",
            "7",
            "resilient",
            "1",
            "--from-correction",
            "--discard-stale-downstream",
            "--cleanup-raw-fits",
        ],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert invocation_log.read_text().strip() == (
        "7 resilient --from-correction --discard-stale-downstream "
        "--cleanup-raw-fits"
    )


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
