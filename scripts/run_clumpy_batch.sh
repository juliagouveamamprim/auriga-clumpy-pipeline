#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_clumpy_batch.sh <start_id> <end_id> <scenario> [max_jobs]
      [--from-correction] [--discard-stale-downstream] [--cleanup-raw-fits]

Arguments:
  start_id    First repopulation ID, inclusive
  end_id      Last repopulation ID, inclusive
  scenario    resilient, fragile, or both
  max_jobs    Maximum number of concurrent cases (default: 8)
  --from-correction
              Reuse existing raw products and rerun only correction onward
  --discard-stale-downstream
              With --from-correction, delete validated stale downstream
              products instead of archiving them
  --cleanup-raw-fits
              Delete only the primary raw CLUMPY FITS after successful pipeline
              completion (default: disabled)

Environment variables:
  POLL_SECONDS          Seconds between concurrency checks (default: 60)
  LAUNCH_DELAY_SECONDS  Delay between launches (default: 5)
  SKIP_COMPLETED        Skip non-empty final total FITS files (default: 1)
  NSIDE                 HEALPix NSIDE inherited by one-case runs (default: 2048)
  EXTENDED_CUT_F        Extended cut factor inherited by one-case runs (default: none)
  POINTLIKE_CUT_F       Pointlike cut factor inherited by one-case runs (default: none)
  FORCE_POINTLIKE       Regenerate valid pointlike FITS files (0 or 1; default: 0)
  MAX_SUBHALO_VMAX_KM_S Inclusive Vmax ceiling in km/s (default: unset/full population)
  CLUMPY_OUTPUT_ROOT     Explicit CLUMPY product root; required for replacement mode
  DARK_INPLACE_REPLACE  Set to 1 to replace case products in CLUMPY_OUTPUT_ROOT
  DARK_INPLACE_REQUIRED_ROOT
                         Required storage prefix for replacement (default: /dados5)
  CLUMPY_EXECUTABLE     CLUMPY executable or wrapper inherited by one-case runs
  PYTHON_EXECUTABLE     Python executable inherited by one-case runs
  CATALOG_ROOT          Root containing repop_XXXX HDF5 catalogues

Examples:
  bash scripts/run_clumpy_batch.sh 160 179 resilient 8
  bash scripts/run_clumpy_batch.sh 160 179 fragile 8
  bash scripts/run_clumpy_batch.sh 160 179 both 8
EOF
}

if [[ $# -lt 3 || $# -gt 7 ]]; then
    usage
    exit 2
fi

START_ID="$1"
END_ID="$2"
SCENARIO_MODE="$3"
MAX_JOBS=8
FROM_CORRECTION=0
MAX_JOBS_SET=0
DISCARD_STALE_DOWNSTREAM=0
CLEANUP_RAW_FITS=0
for argument in "${@:4}"; do
    case "${argument}" in
        --from-correction)
            if [[ "${FROM_CORRECTION}" == "1" ]]; then
                echo "ERROR: duplicate option: ${argument}" >&2
                exit 2
            fi
            FROM_CORRECTION=1
            ;;
        --discard-stale-downstream)
            if [[ "${DISCARD_STALE_DOWNSTREAM}" == "1" ]]; then
                echo "ERROR: duplicate option: ${argument}" >&2
                exit 2
            fi
            DISCARD_STALE_DOWNSTREAM=1
            ;;
        --cleanup-raw-fits)
            if [[ "${CLEANUP_RAW_FITS}" == "1" ]]; then
                echo "ERROR: duplicate option: ${argument}" >&2
                exit 2
            fi
            CLEANUP_RAW_FITS=1
            ;;
        *)
            if [[ "${MAX_JOBS_SET}" == "1" ]]; then
                echo "ERROR: unexpected argument: ${argument}" >&2
                exit 2
            fi
            MAX_JOBS="${argument}"
            MAX_JOBS_SET=1
            ;;
    esac
done

if [[ "${DISCARD_STALE_DOWNSTREAM}" == "1" \
    && "${FROM_CORRECTION}" != "1" ]]; then
    echo "ERROR: --discard-stale-downstream requires --from-correction" >&2
    exit 2
fi

POLL_SECONDS="${POLL_SECONDS:-60}"
LAUNCH_DELAY_SECONDS="${LAUNCH_DELAY_SECONDS:-5}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DEFAULT_NSIDE=2048
NSIDE="${NSIDE:-${DEFAULT_NSIDE}}"
EXTENDED_CUT_F="${EXTENDED_CUT_F:-}"
POINTLIKE_CUT_F="${POINTLIKE_CUT_F:-}"
FORCE_POINTLIKE="${FORCE_POINTLIKE:-0}"
MAX_SUBHALO_VMAX_KM_S="${MAX_SUBHALO_VMAX_KM_S:-}"
CATALOG_ROOT="${CATALOG_ROOT:-}"
CLUMPY_OUTPUT_ROOT_EXPLICIT=0
if [[ -n "${CLUMPY_OUTPUT_ROOT+x}" ]]; then
    CLUMPY_OUTPUT_ROOT_EXPLICIT=1
fi
DARK_INPLACE_REPLACE="${DARK_INPLACE_REPLACE:-0}"
DARK_INPLACE_REQUIRED_ROOT="${DARK_INPLACE_REQUIRED_ROOT:-/dados5}"

export NSIDE EXTENDED_CUT_F POINTLIKE_CUT_F FORCE_POINTLIKE MAX_SUBHALO_VMAX_KM_S
export DARK_INPLACE_REPLACE DARK_INPLACE_REQUIRED_ROOT

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

if ! [[ "${NSIDE}" =~ ^[0-9]+$ ]] || (( NSIDE < 1 || (NSIDE & (NSIDE - 1)) != 0 )); then
    echo "ERROR: NSIDE must be a positive power of two." >&2
    exit 2
fi

if [[ "${FORCE_POINTLIKE}" != "0" && "${FORCE_POINTLIKE}" != "1" ]]; then
    echo "ERROR: FORCE_POINTLIKE must be 0 or 1." >&2
    exit 2
fi

if [[ "${DARK_INPLACE_REPLACE}" != "0" \
    && "${DARK_INPLACE_REPLACE}" != "1" ]]; then
    echo "ERROR: DARK_INPLACE_REPLACE must be 0 or 1." >&2
    exit 2
fi

if [[ -n "${MAX_SUBHALO_VMAX_KM_S}" ]]; then
    if ! [[ "${MAX_SUBHALO_VMAX_KM_S}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || [[ "${MAX_SUBHALO_VMAX_KM_S}" =~ ^0+([.]0+)?$ ]]; then
        echo "ERROR: MAX_SUBHALO_VMAX_KM_S must be a positive decimal number." >&2
        exit 2
    fi
    VMAX_PATH_VALUE=$(printf "%s" "${MAX_SUBHALO_VMAX_KM_S}" \
        | sed -E 's/^0+([0-9])/\1/; s/0+$//; s/[.]$//; s/[.]/p/')
    POPULATION_TAG="dark_vmax_le_${VMAX_PATH_VALUE}kms"
else
    POPULATION_TAG="full"
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
CATALOG_ROOT="${CATALOG_ROOT:-${REPO_ROOT}/outputs}"
CLUMPY_OUTPUT_ROOT="${CLUMPY_OUTPUT_ROOT:-${REPO_ROOT}/outputs/clumpy}"
export CATALOG_ROOT CLUMPY_OUTPUT_ROOT

if [[ "${DARK_INPLACE_REPLACE}" == "1" ]]; then
    if [[ "${FROM_CORRECTION}" == "1" ]]; then
        echo "ERROR: in-place dark replacement requires complete normal runs." >&2
        exit 2
    fi
    if [[ "${CLUMPY_OUTPUT_ROOT_EXPLICIT}" != "1" ]]; then
        echo "ERROR: in-place dark replacement requires explicit CLUMPY_OUTPUT_ROOT." >&2
        exit 2
    fi
    if [[ "${MAX_SUBHALO_VMAX_KM_S}" != "8" \
        || "${EXTENDED_CUT_F}" != "1e-3" \
        || "${POINTLIKE_CUT_F}" != "1e-3" ]]; then
        echo "ERROR: in-place replacement is restricted to Vmax<=8 and both cuts=1e-3." >&2
        exit 2
    fi
    if [[ "${CLUMPY_OUTPUT_ROOT}" != /* \
        || "${DARK_INPLACE_REQUIRED_ROOT}" != /* ]]; then
        echo "ERROR: replacement output and required roots must be absolute paths." >&2
        exit 2
    fi
    if [[ ! -d "${CLUMPY_OUTPUT_ROOT}" \
        || ! -d "${DARK_INPLACE_REQUIRED_ROOT}" ]]; then
        echo "ERROR: replacement output and required roots must already exist." >&2
        exit 2
    fi
    CLUMPY_OUTPUT_ROOT="$(realpath -e -- "${CLUMPY_OUTPUT_ROOT}")"
    DARK_INPLACE_REQUIRED_ROOT="$(realpath -e -- "${DARK_INPLACE_REQUIRED_ROOT}")"
    if [[ "${CLUMPY_OUTPUT_ROOT}" != "${DARK_INPLACE_REQUIRED_ROOT}" \
        && "${CLUMPY_OUTPUT_ROOT}" != "${DARK_INPLACE_REQUIRED_ROOT}/"* ]]; then
        echo "ERROR: replacement output root is outside the required storage root." >&2
        exit 2
    fi
    export CLUMPY_OUTPUT_ROOT DARK_INPLACE_REQUIRED_ROOT
fi

if [[ ! -f "${ONE_CASE_SCRIPT}" ]]; then
    echo "ERROR: one-case wrapper not found: ${ONE_CASE_SCRIPT}" >&2
    exit 1
fi

printf -v START_TAG "%04d" "${START_ID}"
printf -v END_TAG "%04d" "${END_ID}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ "${DARK_INPLACE_REPLACE}" == "1" ]]; then
    POPULATION_RUN_ROOT="${CLUMPY_OUTPUT_ROOT}"
elif [[ "${POPULATION_TAG}" == "full" ]]; then
    POPULATION_RUN_ROOT="${CLUMPY_OUTPUT_ROOT}"
else
    POPULATION_RUN_ROOT="${CLUMPY_OUTPUT_ROOT}/${POPULATION_TAG}"
fi
BATCH_LOG_DIR="${POPULATION_RUN_ROOT}/logs/batches/${SCENARIO_MODE}_${START_TAG}_${END_TAG}_${TIMESTAMP}"
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
echo "From correction:  ${FROM_CORRECTION}"
echo "Discard stale:    ${DISCARD_STALE_DOWNSTREAM}"
echo "Cleanup raw FITS: ${CLEANUP_RAW_FITS}"
echo "Catalogue root:   ${CATALOG_ROOT}"
echo "NSIDE:            ${NSIDE}"
echo "Population:       ${POPULATION_TAG}"
echo "Population root:  ${POPULATION_RUN_ROOT}"
echo "In-place replace: ${DARK_INPLACE_REPLACE}"
echo "Cuts:"
echo "  EXTENDED_CUT_F:      ${EXTENDED_CUT_F:-none}"
echo "  POINTLIKE_CUT_F:     ${POINTLIKE_CUT_F:-none}"
echo "  FORCE_POINTLIKE:     ${FORCE_POINTLIKE}"
echo "  MAX_SUBHALO_VMAX_KM_S: ${MAX_SUBHALO_VMAX_KM_S:-none (full population)}"
echo "  Extended aperture:   hp.max_pixrad(NSIDE), derived automatically"
echo "One-case script:  ${ONE_CASE_SCRIPT}"
echo "Batch log dir:    ${BATCH_LOG_DIR}"
echo "================================================================================"

missing_inputs=0

for repop_id in $(seq "${START_ID}" "${END_ID}"); do
    printf -v repop_tag "repop_%04d" "${repop_id}"

    for scenario in "${SCENARIOS[@]}"; do
        input_h5="${CATALOG_ROOT}/${repop_tag}/fullrepop_hydro_${scenario}.h5"

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

dark_completion_marker_valid() {
    local marker="$1"
    local repop_id="$2"
    local scenario="$3"
    local final_fits="$4"

    [[ -s "${marker}" ]] || return 1
    grep -Fxq "FORMAT=auriga_dark_replacement_v1" "${marker}" \
        && grep -Fxq "REPOP_ID=${repop_id}" "${marker}" \
        && grep -Fxq "SCENARIO=${scenario}" "${marker}" \
        && grep -Fxq "NSIDE=${NSIDE}" "${marker}" \
        && grep -Fxq "MAX_SUBHALO_VMAX_KM_S=8" "${marker}" \
        && grep -Fxq "EXTENDED_CUT_F=1e-3" "${marker}" \
        && grep -Fxq "POINTLIKE_CUT_F=1e-3" "${marker}" \
        && grep -Fxq "TOTAL_FITS=${final_fits}" "${marker}"
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

        ONE_CASE_ARGS=("${repop_id}" "${scenario}")
        if [[ "${FROM_CORRECTION}" == "1" ]]; then
            ONE_CASE_ARGS+=("--from-correction")
            if [[ "${DISCARD_STALE_DOWNSTREAM}" == "1" ]]; then
                ONE_CASE_ARGS+=("--discard-stale-downstream")
            fi
        fi
        if [[ "${CLEANUP_RAW_FITS}" == "1" ]]; then
            ONE_CASE_ARGS+=("--cleanup-raw-fits")
        fi
        bash "${ONE_CASE_SCRIPT}" "${ONE_CASE_ARGS[@]}"
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
        final_fits="${POPULATION_RUN_ROOT}/${scenario}/outputs/total/${repop_tag}_nside${NSIDE}/auriga_total_nside${NSIDE}.fits"
        completion_marker="${POPULATION_RUN_ROOT}/${scenario}/metadata/dark_replacement/${repop_tag}_nside${NSIDE}.complete"

        completed_for_this_run=0
        if [[ "${DARK_INPLACE_REPLACE}" == "1" ]]; then
            if [[ -s "${final_fits}" ]] \
                && dark_completion_marker_valid \
                    "${completion_marker}" "${repop_id}" "${scenario}" \
                    "${final_fits}"; then
                completed_for_this_run=1
            fi
        elif [[ -s "${final_fits}" ]]; then
            completed_for_this_run=1
        fi

        if [[ "${FROM_CORRECTION}" == "0" \
            && "${SKIP_COMPLETED}" == "1" \
            && "${completed_for_this_run}" == "1" ]]; then
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
