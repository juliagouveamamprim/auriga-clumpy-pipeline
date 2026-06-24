#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_clumpy_batch.sh <start_id> <end_id> <scenario> [max_jobs]

Arguments:
  start_id    First repopulation ID, inclusive
  end_id      Last repopulation ID, inclusive
  scenario    resilient, fragile, or both
  max_jobs    Maximum number of concurrent cases (default: 8)

Environment variables:
  POLL_SECONDS          Seconds between concurrency checks (default: 60)
  LAUNCH_DELAY_SECONDS  Delay between launches (default: 5)
  SKIP_COMPLETED        Skip non-empty final total FITS files (default: 1)
  CLUMPY_EXECUTABLE     CLUMPY executable or wrapper inherited by one-case runs
  PYTHON_EXECUTABLE     Python executable inherited by one-case runs

Examples:
  bash scripts/run_clumpy_batch.sh 160 179 resilient 8
  bash scripts/run_clumpy_batch.sh 160 179 fragile 8
  bash scripts/run_clumpy_batch.sh 160 179 both 8
EOF
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
    usage
    exit 2
fi

START_ID="$1"
END_ID="$2"
SCENARIO_MODE="$3"
MAX_JOBS="${4:-8}"

POLL_SECONDS="${POLL_SECONDS:-60}"
LAUNCH_DELAY_SECONDS="${LAUNCH_DELAY_SECONDS:-5}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

for value_name in START_ID END_ID MAX_JOBS POLL_SECONDS LAUNCH_DELAY_SECONDS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${value_name} must be a non-negative integer." >&2
        exit 2
    fi
done

if (( START_ID > END_ID )); then
    echo "ERROR: start_id must be less than or equal to end_id." >&2
    exit 2
fi

if (( MAX_JOBS < 1 )); then
    echo "ERROR: max_jobs must be at least 1." >&2
    exit 2
fi

case "${SCENARIO_MODE}" in
    resilient)
        SCENARIOS=(resilient)
        ;;
    fragile)
        SCENARIOS=(fragile)
        ;;
    both)
        SCENARIOS=(resilient fragile)
        ;;
    *)
        echo "ERROR: scenario must be 'resilient', 'fragile', or 'both'." >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ONE_CASE_SCRIPT="${SCRIPT_DIR}/run_clumpy_one_case.sh"

if [[ ! -f "${ONE_CASE_SCRIPT}" ]]; then
    echo "ERROR: one-case wrapper not found: ${ONE_CASE_SCRIPT}" >&2
    exit 1
fi

printf -v START_TAG "%04d" "${START_ID}"
printf -v END_TAG "%04d" "${END_ID}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

BATCH_LOG_DIR="${REPO_ROOT}/outputs/clumpy/logs/batches/${SCENARIO_MODE}_${START_TAG}_${END_TAG}_${TIMESTAMP}"
mkdir -p "${BATCH_LOG_DIR}"

echo "================================================================================"
echo "Auriga + CLUMPY batch"
echo "================================================================================"
echo "Repository:       ${REPO_ROOT}"
echo "Start ID:         ${START_ID}"
echo "End ID:           ${END_ID}"
echo "Scenario mode:    ${SCENARIO_MODE}"
echo "Maximum jobs:     ${MAX_JOBS}"
echo "Poll interval:    ${POLL_SECONDS} s"
echo "Launch delay:     ${LAUNCH_DELAY_SECONDS} s"
echo "Skip completed:   ${SKIP_COMPLETED}"
echo "One-case script:  ${ONE_CASE_SCRIPT}"
echo "Batch log dir:    ${BATCH_LOG_DIR}"
echo "================================================================================"

missing_inputs=0

for repop_id in $(seq "${START_ID}" "${END_ID}"); do
    printf -v repop_tag "repop_%04d" "${repop_id}"

    for scenario in "${SCENARIOS[@]}"; do
        input_h5="${REPO_ROOT}/outputs/${repop_tag}/fullrepop_hydro_${scenario}.h5"

        if [[ ! -s "${input_h5}" ]]; then
            echo "MISSING INPUT: ${input_h5}" >&2
            missing_inputs=$((missing_inputs + 1))
        fi
    done
done

if (( missing_inputs > 0 )); then
    echo >&2
    echo "ERROR: ${missing_inputs} required HDF5 input(s) are missing or empty." >&2
    echo "The batch was not started." >&2
    exit 1
fi

running_jobs() {
    jobs -rp | wc -l | tr -d ' '
}

declare -a PIDS=()
declare -A LABEL_BY_PID=()
declare -A LOG_BY_PID=()

launched=0
skipped=0

launch_case() {
    local repop_id="$1"
    local scenario="$2"
    local repop_tag
    local log

    printf -v repop_tag "repop_%04d" "${repop_id}"
    log="${BATCH_LOG_DIR}/${repop_tag}_${scenario}.batch.log"

    echo "Launching ${repop_tag} ${scenario}"
    echo "  log: ${log}"

    (
        set +e

        echo "================================================================================"
        echo "START ${repop_tag} ${scenario}"
        echo "Date: $(date)"
        echo "Host: $(hostname)"
        echo "================================================================================"
        echo

        bash "${ONE_CASE_SCRIPT}" "${repop_id}" "${scenario}"
        status=$?

        echo
        echo "================================================================================"
        echo "END ${repop_tag} ${scenario}"
        echo "Date: $(date)"
        echo "Exit status: ${status}"
        echo "================================================================================"

        exit "${status}"
    ) > "${log}" 2>&1 &

    pid=$!
    PIDS+=("${pid}")
    LABEL_BY_PID["${pid}"]="${repop_tag} ${scenario}"
    LOG_BY_PID["${pid}"]="${log}"
    launched=$((launched + 1))
}

terminate_children() {
    trap - INT TERM

    echo
    echo "Interrupt received; terminating active batch jobs..." >&2

    active_pids="$(jobs -rp || true)"
    if [[ -n "${active_pids}" ]]; then
        kill ${active_pids} 2>/dev/null || true
    fi

    wait || true
    exit 130
}

trap terminate_children INT TERM

for repop_id in $(seq "${START_ID}" "${END_ID}"); do
    printf -v repop_tag "repop_%04d" "${repop_id}"

    for scenario in "${SCENARIOS[@]}"; do
        final_fits="${REPO_ROOT}/outputs/clumpy/${scenario}/outputs/total/${repop_tag}/auriga_total_nside1024.fits"

        if [[ "${SKIP_COMPLETED}" == "1" && -s "${final_fits}" ]]; then
            echo "Skipping completed ${repop_tag} ${scenario}"
            echo "  final FITS: ${final_fits}"
            skipped=$((skipped + 1))
            continue
        fi

        while (( $(running_jobs) >= MAX_JOBS )); do
            sleep "${POLL_SECONDS}"
        done

        launch_case "${repop_id}" "${scenario}"

        if (( LAUNCH_DELAY_SECONDS > 0 )); then
            sleep "${LAUNCH_DELAY_SECONDS}"
        fi
    done
done

echo
echo "All selected cases have been launched. Waiting for completion..."

succeeded=0
failed=0

for pid in "${PIDS[@]}"; do
    if wait "${pid}"; then
        succeeded=$((succeeded + 1))
    else
        failed=$((failed + 1))
        echo "FAILED: ${LABEL_BY_PID[${pid}]}" >&2
        echo "  log: ${LOG_BY_PID[${pid}]}" >&2
    fi
done

echo
echo "================================================================================"
echo "Batch finished"
echo "================================================================================"
echo "Date:       $(date)"
echo "Launched:   ${launched}"
echo "Succeeded:  ${succeeded}"
echo "Failed:     ${failed}"
echo "Skipped:    ${skipped}"
echo "Log dir:    ${BATCH_LOG_DIR}"
echo "================================================================================"

if (( failed > 0 )); then
    exit 1
fi
