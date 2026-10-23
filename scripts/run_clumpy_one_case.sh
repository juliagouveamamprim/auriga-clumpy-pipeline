#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run one complete CLUMPY pipeline case
# ============================================================
#
# Usage:
#
#   bash scripts/run_clumpy_one_case.sh <repop_id> <scenario> [--from-correction]
#       [--discard-stale-downstream] [--cleanup-raw-fits]
#
# Examples:
#
#   bash scripts/run_clumpy_one_case.sh 1 resilient
#   bash scripts/run_clumpy_one_case.sh 1 fragile
#
# Pipeline:
#
#   1. HDF5 -> extended CLUMPY list + pointlike HEALPix FITS
#   2. generate raw CLUMPY params
#   3. run CLUMPY raw
#   4. correct rho_s from raw halo_rendered.log
#   5. generate corrected CLUMPY params
#   6. run CLUMPY corrected
#   7. combine corrected CLUMPY and pointlike FITS
#
# Not included:
#   - diagnostics/diagnostic_Jfac.py
#   - plot_extended.py
# ============================================================

if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
    echo "Usage:"
    echo "  bash $0 <repop_id> <scenario> [--from-correction] [--discard-stale-downstream] [--cleanup-raw-fits]"
    echo
    echo "Examples:"
    echo "  bash $0 1 resilient"
    echo "  bash $0 1 fragile"
    exit 1
fi

REPOP_ID="$1"
SCENARIO="$2"
RUN_MODE="normal"
DISCARD_STALE_DOWNSTREAM=0
CLEANUP_RAW_FITS=0
for option in "${@:3}"; do
    case "${option}" in
        --from-correction)
            if [ "${RUN_MODE}" = "from-correction" ]; then
                echo "ERROR: duplicate option: ${option}" >&2
                exit 2
            fi
            RUN_MODE="from-correction"
            ;;
        --discard-stale-downstream)
            if [ "${DISCARD_STALE_DOWNSTREAM}" = "1" ]; then
                echo "ERROR: duplicate option: ${option}" >&2
                exit 2
            fi
            DISCARD_STALE_DOWNSTREAM=1
            ;;
        --cleanup-raw-fits)
            if [ "${CLEANUP_RAW_FITS}" = "1" ]; then
                echo "ERROR: duplicate option: ${option}" >&2
                exit 2
            fi
            CLEANUP_RAW_FITS=1
            ;;
        *)
            echo "ERROR: unknown mode: ${option}" >&2
            exit 2
            ;;
    esac
done

if [ "${DISCARD_STALE_DOWNSTREAM}" = "1" ] \
    && [ "${RUN_MODE}" != "from-correction" ]; then
    echo "ERROR: --discard-stale-downstream requires --from-correction" >&2
    exit 2
fi

if [ "$SCENARIO" != "resilient" ] && [ "$SCENARIO" != "fragile" ]; then
    echo "ERROR: scenario must be 'resilient' or 'fragile'."
    exit 1
fi

if ! [[ "$REPOP_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: repop_id must be a non-negative integer."
    exit 1
fi

REPOP_TAG=$(printf "repop_%04d" "$REPOP_ID")

SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPTS_DIR}/.." && pwd)"
BASE_RUN_DIR="${REPOSITORY_ROOT}/outputs/clumpy"
CATALOG_ROOT="${CATALOG_ROOT:-${REPOSITORY_ROOT}/outputs}"

# Set CLUMPY_EXECUTABLE to the CLUMPY binary or wrapper.
# The executable must produce the patched *.halo_rendered.log output.
CLUMPY="${CLUMPY_EXECUTABLE:-clumpy}"
PYTHON="${PYTHON_EXECUTABLE:-python3}"

DEFAULT_NSIDE=2048
NSIDE="${NSIDE:-${DEFAULT_NSIDE}}"

EXTENDED_CUT_F="${EXTENDED_CUT_F:-}"
POINTLIKE_CUT_F="${POINTLIKE_CUT_F:-}"
FORCE_POINTLIKE="${FORCE_POINTLIKE:-0}"

if ! [[ "${NSIDE}" =~ ^[0-9]+$ ]] || (( NSIDE < 1 || (NSIDE & (NSIDE - 1)) != 0 )); then
    echo "ERROR: NSIDE must be a positive power of two."
    exit 1
fi

if [ "${FORCE_POINTLIKE}" != "0" ] && [ "${FORCE_POINTLIKE}" != "1" ]; then
    echo "ERROR: FORCE_POINTLIKE must be 0 or 1."
    exit 1
fi

NSIDE_SUFFIX="_nside${NSIDE}"

REPOP_RUN_TAG="${REPOP_TAG}${NSIDE_SUFFIX}"
CLUMPY_BASENAME="annihil_gal2D_LOS0_0_FOV360x180_nside${NSIDE}"

if ! command -v "$CLUMPY" >/dev/null 2>&1; then
    echo "ERROR: CLUMPY executable not found:"
    echo "  $CLUMPY"
    echo
    echo "Set it with, for example:"
    echo "  export CLUMPY_EXECUTABLE=/path/to/clumpy_wrapper"
    exit 1
fi

CLUMPY_RESOLVED="$(command -v -- "$CLUMPY")"
CLUMPY_RESOLVED="$(readlink -f -- "$CLUMPY_RESOLVED")"

if [ -z "${CLUMPY_DATA:-}" ]; then
    case "$CLUMPY_RESOLVED" in
        */bin/clumpy)
            CLUMPY_DATA="${CLUMPY_RESOLVED%/bin/clumpy}/data"
            ;;
        */build/cli/clumpy)
            CLUMPY_DATA="${CLUMPY_RESOLVED%/build/cli/clumpy}/data"
            ;;
        *)
            echo "ERROR: CLUMPY_DATA is not set and could not be inferred from:" >&2
            echo "  $CLUMPY_RESOLVED" >&2
            echo >&2
            echo "Set CLUMPY_DATA to the CLUMPY data directory." >&2
            exit 1
            ;;
    esac
fi

if [ ! -d "$CLUMPY_DATA" ] \
    || [ ! -f "$CLUMPY_DATA/list_generic.txt" ] \
    || [ ! -d "$CLUMPY_DATA/healpix" ]; then
    echo "ERROR: CLUMPY_DATA is not a valid CLUMPY data directory:" >&2
    echo "  $CLUMPY_DATA" >&2
    echo >&2
    echo "Expected at least:" >&2
    echo "  $CLUMPY_DATA/list_generic.txt" >&2
    echo "  $CLUMPY_DATA/healpix/" >&2
    exit 1
fi

CLUMPY_DATA="$(cd -- "$CLUMPY_DATA" && pwd -P)"
export CLUMPY_DATA
CLUMPY="$CLUMPY_RESOLVED"

CASE_DIR="${BASE_RUN_DIR}/${SCENARIO}"

RAW_PARAM="${CASE_DIR}/params/generated/${REPOP_TAG}_raw_params${NSIDE_SUFFIX}.txt"
CORRECTED_PARAM="${CASE_DIR}/params/generated/${REPOP_TAG}_corrected_params${NSIDE_SUFFIX}.txt"

RAW_LIST="${CASE_DIR}/lists/raw/${REPOP_TAG}_raw_nopointlike${NSIDE_SUFFIX}.txt"
POINTLIKE_FITS="${CASE_DIR}/pointlike/${REPOP_TAG}_pointlike_nside${NSIDE}.fits"

RAW_CLUMPY_LOG="${CASE_DIR}/logs/raw_clumpy/${REPOP_TAG}_raw_clumpy${NSIDE_SUFFIX}.log"
CORRECTED_CLUMPY_LOG="${CASE_DIR}/logs/corrected_clumpy/${REPOP_TAG}_corrected_clumpy${NSIDE_SUFFIX}.log"

RAW_OUTPUT_DIR="${CASE_DIR}/outputs/raw_clumpy/${REPOP_RUN_TAG}"
CORRECTED_OUTPUT_DIR="${CASE_DIR}/outputs/corrected_clumpy/${REPOP_RUN_TAG}"

RAW_FITS="${RAW_OUTPUT_DIR}/${CLUMPY_BASENAME}.fits"
RAW_RENDERED_LOG="${RAW_OUTPUT_DIR}/${CLUMPY_BASENAME}.halo_rendered.log"
CORRECTED_RENDERED_LOG="${CORRECTED_OUTPUT_DIR}/${CLUMPY_BASENAME}.halo_rendered.log"

FINAL_FITS="${CORRECTED_OUTPUT_DIR}/${CLUMPY_BASENAME}.fits"
TOTAL_OUTPUT_DIR="${CASE_DIR}/outputs/total/${REPOP_TAG}_nside${NSIDE}"
TOTAL_FITS="${TOTAL_OUTPUT_DIR}/auriga_total_nside${NSIDE}.fits"
CATALOG_H5="${CATALOG_ROOT}/${REPOP_TAG}/fullrepop_hydro_${SCENARIO}.h5"
CORRECTED_LIST="${CASE_DIR}/lists/corrected/${REPOP_TAG}_rhocorr${NSIDE_SUFFIX}.txt"
CORRECTION_OUTPUT_DIR="${CASE_DIR}/logs/correction/${REPOP_RUN_TAG}"

cleanup_primary_raw_fits() {
    local base_run_resolved raw_output_resolved expected_raw_output
    local raw_filename cleanup_target

    if [ "${CLEANUP_RAW_FITS}" != "1" ]; then
        return 0
    fi

    base_run_resolved=$(realpath -e -- "${BASE_RUN_DIR}") || {
        echo "ERROR: could not resolve CLUMPY output root for raw FITS cleanup:" >&2
        echo "  ${BASE_RUN_DIR}" >&2
        return 1
    }
    raw_output_resolved=$(realpath -e -- "${RAW_OUTPUT_DIR}") || {
        echo "ERROR: could not resolve raw CLUMPY output directory for cleanup:" >&2
        echo "  ${RAW_OUTPUT_DIR}" >&2
        return 1
    }
    expected_raw_output="${base_run_resolved}/${SCENARIO}/outputs/raw_clumpy/${REPOP_RUN_TAG}"

    if [ "${raw_output_resolved}" != "${expected_raw_output}" ]; then
        echo "ERROR: refusing raw FITS cleanup from an unexpected resolved directory:" >&2
        echo "  requested: ${RAW_OUTPUT_DIR}" >&2
        echo "  resolved:  ${raw_output_resolved}" >&2
        echo "  expected:  ${expected_raw_output}" >&2
        return 1
    fi

    raw_filename=$(basename -- "${RAW_FITS}")
    if [ "$(dirname -- "${RAW_FITS}")" != "${RAW_OUTPUT_DIR}" ] \
        || [ "${raw_filename}" != "${CLUMPY_BASENAME}.fits" ]; then
        echo "ERROR: refusing unexpected raw FITS cleanup target:" >&2
        echo "  ${RAW_FITS}" >&2
        return 1
    fi

    cleanup_target="${raw_output_resolved}/${raw_filename}"
    if [ -L "${cleanup_target}" ]; then
        echo "ERROR: refusing to remove a symlink at the raw FITS path:" >&2
        echo "  ${cleanup_target}" >&2
        return 1
    fi
    if [ ! -e "${cleanup_target}" ]; then
        echo "Raw CLUMPY FITS is already absent; no cleanup needed:"
        echo "  ${cleanup_target}"
        return 0
    fi
    if [ ! -f "${cleanup_target}" ]; then
        echo "ERROR: raw FITS cleanup target is not a regular file:" >&2
        echo "  ${cleanup_target}" >&2
        return 1
    fi

    rm -f -- "${cleanup_target}"
    if [ -e "${cleanup_target}" ] || [ -L "${cleanup_target}" ]; then
        echo "ERROR: raw FITS cleanup target still exists after removal:" >&2
        echo "  ${cleanup_target}" >&2
        return 1
    fi
    echo "Removed primary raw CLUMPY FITS:"
    echo "  ${cleanup_target}"
}

echo
echo "======================================================================"
echo "Starting CLUMPY pipeline one-case run"
echo "======================================================================"
echo "REPOP_ID:  ${REPOP_ID}"
echo "REPOP_TAG: ${REPOP_TAG}"
echo "SCENARIO:  ${SCENARIO}"
echo "NSIDE:     ${NSIDE}"
echo "Run tag:   ${REPOP_RUN_TAG}"
echo "Run mode:  ${RUN_MODE}"
echo "Discard stale downstream: ${DISCARD_STALE_DOWNSTREAM}"
echo "Cleanup raw FITS: ${CLEANUP_RAW_FITS}"
echo "Catalogue root: ${CATALOG_ROOT}"
echo "CLUMPY:    ${CLUMPY}"
echo "CLUMPY_DATA: ${CLUMPY_DATA}"
echo "Cuts:"
echo "  EXTENDED_CUT_F:      ${EXTENDED_CUT_F:-none}"
echo "  POINTLIKE_CUT_F:     ${POINTLIKE_CUT_F:-none}"
echo "  FORCE_POINTLIKE:     ${FORCE_POINTLIKE}"
echo "  Extended aperture:   hp.max_pixrad(NSIDE), derived automatically"
echo "Time:      $(date)"
echo "======================================================================"
echo

if [ "${RUN_MODE}" = "from-correction" ]; then
    echo "----------------------------------------------------------------------"
    echo "Downstream-only rerun from existing raw correction inputs"
    echo "----------------------------------------------------------------------"

    for required_path in \
        "${CATALOG_H5}" \
        "${RAW_LIST}" \
        "${RAW_RENDERED_LOG}" \
        "${POINTLIKE_FITS}"; do
        if [ ! -f "${required_path}" ] || [ ! -s "${required_path}" ]; then
            echo "ERROR: required downstream-rerun input is missing or empty:" >&2
            echo "  ${required_path}" >&2
            exit 1
        fi
    done

    echo "Validating authoritative HDF5 and raw list/log identity (read only)..."
    "${PYTHON}" "${SCRIPTS_DIR}/correct_rhos_from_clumpy_raw.py" \
        "${REPOP_ID}" "${SCENARIO}" \
        --nside "${NSIDE}" \
        --catalog-root "${CATALOG_ROOT}" \
        --clumpy-root "${BASE_RUN_DIR}" \
        --raw-list "${RAW_LIST}" \
        --raw-rendered-log "${RAW_RENDERED_LOG}" \
        --validate-only

    echo "Validating existing point-like FITS (read only)..."
    "${PYTHON}" "${SCRIPTS_DIR}/prepare_subhalo_components.py" \
        "${REPOP_ID}" "${SCENARIO}" \
        --nside "${NSIDE}" \
        --catalog-root "${CATALOG_ROOT}" \
        --validate-pointlike-only

    resolve_product_path() {
        realpath -m -- "$1"
    }

    paths_overlap() {
        local first="$1"
        local second="$2"
        [ "${first}" = "${second}" ] \
            || [[ "${first}" == "${second}/"* ]] \
            || [[ "${second}" == "${first}/"* ]]
    }

    CASE_DIR_RESOLVED=$(resolve_product_path "${CASE_DIR}")
    PRESERVE_PATHS=(
        "${CATALOG_H5}"
        "${RAW_LIST}"
        "${RAW_PARAM}"
        "${RAW_OUTPUT_DIR}"
        "${RAW_RENDERED_LOG}"
        "${RAW_CLUMPY_LOG}"
        "${POINTLIKE_FITS}"
    )
    STALE_DOWNSTREAM_PATHS=(
        "${CORRECTION_OUTPUT_DIR}"
        "${CORRECTED_LIST}"
        "${CORRECTED_PARAM}"
        "${CORRECTED_OUTPUT_DIR}"
        "${CORRECTED_CLUMPY_LOG}"
        "${TOTAL_OUTPUT_DIR}"
    )
    PRESERVE_PATHS_RESOLVED=()
    STALE_DOWNSTREAM_PATHS_RESOLVED=()

    for path in "${PRESERVE_PATHS[@]}"; do
        PRESERVE_PATHS_RESOLVED+=("$(resolve_product_path "${path}")")
    done
    for path in "${STALE_DOWNSTREAM_PATHS[@]}"; do
        resolved_path=$(resolve_product_path "${path}")
        if [[ "${resolved_path}" != "${CASE_DIR_RESOLVED}/"* ]]; then
            echo "ERROR: refusing downstream cleanup path outside case tree:" >&2
            echo "  ${resolved_path}" >&2
            echo "Expected beneath:" >&2
            echo "  ${CASE_DIR_RESOLVED}" >&2
            exit 1
        fi
        for preserve_path in "${PRESERVE_PATHS_RESOLVED[@]}"; do
            if paths_overlap "${resolved_path}" "${preserve_path}"; then
                echo "ERROR: downstream cleanup path overlaps a preserved path:" >&2
                echo "  remove:   ${resolved_path}" >&2
                echo "  preserve: ${preserve_path}" >&2
                exit 1
            fi
        done
        STALE_DOWNSTREAM_PATHS_RESOLVED+=("${resolved_path}")
    done

    if [ "${DISCARD_STALE_DOWNSTREAM}" = "1" ]; then
        echo
        echo "WILL PRESERVE:"
        for path in "${PRESERVE_PATHS_RESOLVED[@]}"; do
            echo "  ${path}"
        done
        echo "WILL REMOVE:"
        for path in "${STALE_DOWNSTREAM_PATHS_RESOLVED[@]}"; do
            echo "  ${path}"
        done
        echo
        echo "All read-only prerequisites passed; removing stale downstream products."
        for path in "${STALE_DOWNSTREAM_PATHS_RESOLVED[@]}"; do
            if [ -e "${path}" ] || [ -L "${path}" ]; then
                rm -rf -- "${path}"
            fi
        done
    fi

    mkdir -p "${CASE_DIR}"
    BACKFILL_STAGE_ROOT=$(mktemp -d \
        "${CASE_DIR}/.from_correction_${REPOP_RUN_TAG}.XXXXXX")
    STAGE_CORRECTION_DIR="${BACKFILL_STAGE_ROOT}/logs/correction/${REPOP_RUN_TAG}"
    STAGE_CORRECTED_LIST="${BACKFILL_STAGE_ROOT}/lists/corrected/${REPOP_TAG}_rhocorr${NSIDE_SUFFIX}.txt"
    STAGE_CORRECTED_PARAM="${BACKFILL_STAGE_ROOT}/params/${REPOP_TAG}_corrected_params${NSIDE_SUFFIX}.txt"
    STAGE_STABLE_CORRECTED_PARAM="${BACKFILL_STAGE_ROOT}/stable_params/${REPOP_TAG}_corrected_params${NSIDE_SUFFIX}.txt"
    STAGE_CORRECTED_LOG="${BACKFILL_STAGE_ROOT}/logs/corrected_clumpy/${REPOP_TAG}_corrected_clumpy${NSIDE_SUFFIX}.log"
    STAGE_CORRECTED_OUTPUT_DIR="${BACKFILL_STAGE_ROOT}/outputs/corrected_clumpy/${REPOP_RUN_TAG}"
    STAGE_CORRECTED_FITS="${STAGE_CORRECTED_OUTPUT_DIR}/${CLUMPY_BASENAME}.fits"
    STAGE_CORRECTED_RENDERED_LOG="${STAGE_CORRECTED_OUTPUT_DIR}/${CLUMPY_BASENAME}.halo_rendered.log"
    STAGE_TOTAL_OUTPUT_DIR="${BACKFILL_STAGE_ROOT}/outputs/total/${REPOP_RUN_TAG}"
    STAGE_TOTAL_FITS="${STAGE_TOTAL_OUTPUT_DIR}/auriga_total_nside${NSIDE}.fits"

    echo "Staging downstream products under:"
    echo "  ${BACKFILL_STAGE_ROOT}"

    "${PYTHON}" "${SCRIPTS_DIR}/correct_rhos_from_clumpy_raw.py" \
        "${REPOP_ID}" "${SCENARIO}" \
        --nside "${NSIDE}" \
        --catalog-root "${CATALOG_ROOT}" \
        --clumpy-root "${BASE_RUN_DIR}" \
        --raw-list "${RAW_LIST}" \
        --raw-rendered-log "${RAW_RENDERED_LOG}" \
        --output-list "${STAGE_CORRECTED_LIST}" \
        --output-dir "${STAGE_CORRECTION_DIR}" \
        --metadata-output-dir "${CORRECTION_OUTPUT_DIR}"

    if [ ! -s "${STAGE_CORRECTED_LIST}" ]; then
        echo "ERROR: staged corrected list was not created or is empty" >&2
        exit 1
    fi

    "${PYTHON}" "${SCRIPTS_DIR}/generate_clumpy_params.py" \
        "${REPOP_ID}" "${SCENARIO}" corrected \
        --nside "${NSIDE}" \
        --input-list "${STAGE_CORRECTED_LIST}" \
        --output-dir "${STAGE_CORRECTED_OUTPUT_DIR}" \
        --output-param "${STAGE_CORRECTED_PARAM}"

    if [ ! -s "${STAGE_CORRECTED_PARAM}" ]; then
        echo "ERROR: staged corrected parameter file was not created or is empty" >&2
        exit 1
    fi

    "${PYTHON}" "${SCRIPTS_DIR}/generate_clumpy_params.py" \
        "${REPOP_ID}" "${SCENARIO}" corrected \
        --nside "${NSIDE}" \
        --input-list "${CORRECTED_LIST}" \
        --validation-input-list "${STAGE_CORRECTED_LIST}" \
        --output-dir "${CORRECTED_OUTPUT_DIR}" \
        --output-param "${STAGE_STABLE_CORRECTED_PARAM}" \
        --defer-output-dir-creation

    mkdir -p "$(dirname "${STAGE_CORRECTED_LOG}")"
    /usr/bin/time -v "${CLUMPY}" -g6 -i "${STAGE_CORRECTED_PARAM}" \
        2>&1 | tee "${STAGE_CORRECTED_LOG}"

    if [ ! -s "${STAGE_CORRECTED_RENDERED_LOG}" ]; then
        echo "ERROR: staged corrected halo_rendered.log is missing or empty" >&2
        exit 1
    fi
    if [ ! -s "${STAGE_CORRECTED_FITS}" ]; then
        echo "ERROR: staged corrected FITS is missing or empty" >&2
        exit 1
    fi

    "${PYTHON}" "${SCRIPTS_DIR}/combine_clumpy_pointlike.py" \
        "${REPOP_ID}" "${SCENARIO}" \
        --nside "${NSIDE}" \
        --corrected-fits "${STAGE_CORRECTED_FITS}" \
        --pointlike-fits "${POINTLIKE_FITS}" \
        --output-fits "${STAGE_TOTAL_FITS}"

    if [ ! -s "${STAGE_TOTAL_FITS}" ]; then
        echo "ERROR: staged total FITS is missing or empty" >&2
        exit 1
    fi

    BACKUP_ROOT=""
    if [ "${DISCARD_STALE_DOWNSTREAM}" = "0" ]; then
        BACKUP_ROOT="${CASE_DIR}/backups/from_correction/${REPOP_RUN_TAG}_$(date +%Y%m%d_%H%M%S)_$$"
        mkdir -p "${BACKUP_ROOT}"
    fi
    PROMOTION_ACTIVE=1
    PROMOTED_DESTINATIONS=()
    PROMOTED_BACKUPS=()
    PROMOTED_LABELS=()

    rollback_promotions() {
        local index destination backup label failed_dir
        failed_dir="${BACKFILL_STAGE_ROOT}/failed_promotion"
        mkdir -p "${failed_dir}"
        for ((index=${#PROMOTED_DESTINATIONS[@]}-1; index>=0; index--)); do
            destination="${PROMOTED_DESTINATIONS[index]}"
            backup="${PROMOTED_BACKUPS[index]}"
            label="${PROMOTED_LABELS[index]}"
            if [ -e "${destination}" ]; then
                mv "${destination}" "${failed_dir}/${label}"
            fi
            if [ -n "${backup}" ] && [ -e "${backup}" ]; then
                mkdir -p "$(dirname "${destination}")"
                mv "${backup}" "${destination}"
            fi
        done
    }

    promotion_error() {
        local status=$?
        if [ "${PROMOTION_ACTIVE}" = "1" ]; then
            if [ "${DISCARD_STALE_DOWNSTREAM}" = "1" ]; then
                echo "ERROR: downstream promotion failed; returning new products to staging" >&2
            else
                echo "ERROR: downstream promotion failed; restoring prior stable products" >&2
            fi
            PROMOTION_ACTIVE=0
            rollback_promotions
        fi
        echo "Staged products retained at: ${BACKFILL_STAGE_ROOT}" >&2
        exit "${status}"
    }
    trap promotion_error ERR

    promote_product() {
        local source="$1"
        local destination="$2"
        local label="$3"
        local backup=""
        mkdir -p "$(dirname "${destination}")"
        if [ -e "${destination}" ]; then
            if [ "${DISCARD_STALE_DOWNSTREAM}" = "1" ]; then
                echo "ERROR: downstream destination unexpectedly exists after cleanup:" >&2
                echo "  ${destination}" >&2
                return 1
            else
                backup="${BACKUP_ROOT}/${label}"
                mkdir -p "$(dirname "${backup}")"
                mv "${destination}" "${backup}"
            fi
        fi
        if ! mv "${source}" "${destination}"; then
            if [ -n "${backup}" ] && [ -e "${backup}" ]; then
                mv "${backup}" "${destination}"
            fi
            return 1
        fi
        PROMOTED_DESTINATIONS+=("${destination}")
        PROMOTED_BACKUPS+=("${backup}")
        PROMOTED_LABELS+=("${label}")
    }

    promote_product "${STAGE_CORRECTION_DIR}" "${CORRECTION_OUTPUT_DIR}" \
        "correction_diagnostics"
    promote_product "${STAGE_CORRECTED_LIST}" "${CORRECTED_LIST}" \
        "corrected_list.txt"
    promote_product "${STAGE_STABLE_CORRECTED_PARAM}" "${CORRECTED_PARAM}" \
        "corrected_params.txt"
    promote_product "${STAGE_CORRECTED_OUTPUT_DIR}" "${CORRECTED_OUTPUT_DIR}" \
        "corrected_clumpy_output"
    promote_product "${STAGE_CORRECTED_LOG}" "${CORRECTED_CLUMPY_LOG}" \
        "corrected_clumpy_stdout.log"
    promote_product "${STAGE_TOTAL_OUTPUT_DIR}" "${TOTAL_OUTPUT_DIR}" \
        "total_output"

    if [ "${DISCARD_STALE_DOWNSTREAM}" = "1" ]; then
        rm -f -- "${STAGE_CORRECTED_PARAM}"
        rm -rf -- "${BACKFILL_STAGE_ROOT}"
    else
        mv "${STAGE_CORRECTED_PARAM}" "${BACKUP_ROOT}/staging_execution_params.txt"
    fi
    PROMOTION_ACTIVE=0
    trap - ERR

    cleanup_primary_raw_fits

    echo
    echo "Downstream-only correction rerun completed successfully."
    if [ "${CLEANUP_RAW_FITS}" = "1" ]; then
        echo "Raw correction inputs were reused; the primary raw FITS cleanup was enabled."
    else
        echo "Raw products were read only and were not regenerated."
    fi
    if [ "${DISCARD_STALE_DOWNSTREAM}" = "1" ]; then
        echo "Stale downstream products were discarded after validation; no backup was retained."
    else
        echo "Previous downstream products, when present, were archived under:"
        echo "  ${BACKUP_ROOT}"
    fi
    echo "Stable total FITS:"
    echo "  ${TOTAL_FITS}"
    exit 0
fi

if [ -f "$TOTAL_FITS" ] && [ -s "$TOTAL_FITS" ]; then
    echo "ERROR: final total FITS already exists and is non-empty:"
    echo "  $TOTAL_FITS"
    echo
    echo "Refusing to overwrite an apparently completed run."
    exit 1
fi

if [ -f "$FINAL_FITS" ] && [ -s "$FINAL_FITS" ]; then
    if [ ! -f "$POINTLIKE_FITS" ] || [ ! -s "$POINTLIKE_FITS" ]; then
        echo "ERROR: cannot resume final map combination because the pointlike FITS is missing or empty:"
        echo "  $POINTLIKE_FITS"
        echo
        echo "The existing corrected CLUMPY FITS was left unchanged:"
        echo "  $FINAL_FITS"
        exit 1
    fi

    echo "Resuming from existing corrected CLUMPY and pointlike FITS files."
    echo "  Corrected CLUMPY: $FINAL_FITS"
    echo "  Pointlike:        $POINTLIKE_FITS"

    "${PYTHON}" "${SCRIPTS_DIR}/combine_clumpy_pointlike.py" \
        "${REPOP_ID}" "${SCENARIO}" --nside "${NSIDE}"

    if [ ! -f "$TOTAL_FITS" ] || [ ! -s "$TOTAL_FITS" ]; then
        echo "ERROR: total FITS was not created or is empty:"
        echo "  $TOTAL_FITS"
        exit 1
    fi

    cleanup_primary_raw_fits

    echo "Final total FITS:"
    echo "  $TOTAL_FITS"
    exit 0
fi

# ============================================================
# 1. HDF5 -> raw CLUMPY list without pointlikes
# ============================================================

echo
echo "----------------------------------------------------------------------"
echo "Step 1/7: prepare extended list and pointlike HEALPix map"
echo "----------------------------------------------------------------------"

PREPARE_ARGS=(
    "${REPOP_ID}"
    "${SCENARIO}"
    "--nside" "${NSIDE}"
    "--catalog-root" "${CATALOG_ROOT}"
)

if [ -n "${EXTENDED_CUT_F}" ]; then
    PREPARE_ARGS+=("--extended-cut-f" "${EXTENDED_CUT_F}")
fi

if [ -n "${POINTLIKE_CUT_F}" ]; then
    PREPARE_ARGS+=("--pointlike-cut-f" "${POINTLIKE_CUT_F}")
fi

if [ "${FORCE_POINTLIKE}" = "1" ]; then
    PREPARE_ARGS+=("--force-pointlike")
fi

"${PYTHON}" "${SCRIPTS_DIR}/prepare_subhalo_components.py" "${PREPARE_ARGS[@]}"

if [ ! -f "$RAW_LIST" ]; then
    echo "ERROR: raw CLUMPY list was not created:"
    echo "  $RAW_LIST"
    exit 1
fi

if [ ! -f "$POINTLIKE_FITS" ] || [ ! -s "$POINTLIKE_FITS" ]; then
    echo "ERROR: pointlike FITS was not created or is empty:"
    echo "  $POINTLIKE_FITS"
    exit 1
fi

# ============================================================
# 2. Generate raw CLUMPY params
# ============================================================

echo
echo "----------------------------------------------------------------------"
echo "Step 2/7: generate raw CLUMPY parameter file"
echo "----------------------------------------------------------------------"

"${PYTHON}" "${SCRIPTS_DIR}/generate_clumpy_params.py" \
    "${REPOP_ID}" "${SCENARIO}" raw --nside "${NSIDE}"

if [ ! -f "$RAW_PARAM" ]; then
    echo "ERROR: raw parameter file was not created:"
    echo "  $RAW_PARAM"
    exit 1
fi

# ============================================================
# 3. Run CLUMPY raw
# ============================================================

echo
echo "----------------------------------------------------------------------"
echo "Step 3/7: run CLUMPY raw"
echo "----------------------------------------------------------------------"

mkdir -p "$(dirname "${RAW_CLUMPY_LOG}")"

/usr/bin/time -v "${CLUMPY}" -g6 -i "${RAW_PARAM}" \
    2>&1 | tee "${RAW_CLUMPY_LOG}"

if [ ! -f "${RAW_RENDERED_LOG}" ]; then
    echo "ERROR: raw halo_rendered.log was not created:"
    echo "  ${RAW_RENDERED_LOG}"
    exit 1
fi

# ============================================================
# 4. Correct rho_s
# ============================================================

echo
echo "----------------------------------------------------------------------"
echo "Step 4/7: correct rho_s from raw CLUMPY rendered log"
echo "----------------------------------------------------------------------"

"${PYTHON}" "${SCRIPTS_DIR}/correct_rhos_from_clumpy_raw.py" \
    "${REPOP_ID}" "${SCENARIO}" --nside "${NSIDE}" \
    --catalog-root "${CATALOG_ROOT}" --clumpy-root "${BASE_RUN_DIR}"

if [ ! -f "$CORRECTED_LIST" ]; then
    echo "ERROR: corrected list was not created:"
    echo "  $CORRECTED_LIST"
    exit 1
fi

# ============================================================
# 5. Generate corrected CLUMPY params
# ============================================================

echo
echo "----------------------------------------------------------------------"
echo "Step 5/7: generate corrected CLUMPY parameter file"
echo "----------------------------------------------------------------------"

"${PYTHON}" "${SCRIPTS_DIR}/generate_clumpy_params.py" \
    "${REPOP_ID}" "${SCENARIO}" corrected --nside "${NSIDE}"

if [ ! -f "$CORRECTED_PARAM" ]; then
    echo "ERROR: corrected parameter file was not created:"
    echo "  $CORRECTED_PARAM"
    exit 1
fi

# ============================================================
# 6. Run CLUMPY corrected
# ============================================================

echo
echo "----------------------------------------------------------------------"
echo "Step 6/7: run CLUMPY corrected"
echo "----------------------------------------------------------------------"

mkdir -p "$(dirname "${CORRECTED_CLUMPY_LOG}")"

/usr/bin/time -v "${CLUMPY}" -g6 -i "${CORRECTED_PARAM}" \
    2>&1 | tee "${CORRECTED_CLUMPY_LOG}"

if [ ! -f "${CORRECTED_RENDERED_LOG}" ]; then
    echo "ERROR: corrected halo_rendered.log was not created:"
    echo "  ${CORRECTED_RENDERED_LOG}"
    exit 1
fi

if [ ! -f "$FINAL_FITS" ] || [ ! -s "$FINAL_FITS" ]; then
    echo "ERROR: final corrected FITS was not created or is empty:"
    echo "  $FINAL_FITS"
    exit 1
fi

# ============================================================
# 7. Combine corrected CLUMPY and pointlike maps
# ============================================================

echo
echo "----------------------------------------------------------------------"
echo "Step 7/7: combine corrected CLUMPY and pointlike FITS"
echo "----------------------------------------------------------------------"

"${PYTHON}" "${SCRIPTS_DIR}/combine_clumpy_pointlike.py" \
    "${REPOP_ID}" "${SCENARIO}" --nside "${NSIDE}"

if [ ! -f "$TOTAL_FITS" ] || [ ! -s "$TOTAL_FITS" ]; then
    echo "ERROR: total FITS was not created or is empty:"
    echo "  $TOTAL_FITS"
    exit 1
fi

cleanup_primary_raw_fits

echo
echo "======================================================================"
echo "Finished CLUMPY pipeline one-case run successfully"
echo "======================================================================"
echo "REPOP_TAG: ${REPOP_TAG}"
echo "SCENARIO:  ${SCENARIO}"
echo "Time:      $(date)"
echo
echo "Raw output:"
echo "  ${RAW_OUTPUT_DIR}"
echo
echo "Corrected output:"
echo "  ${CORRECTED_OUTPUT_DIR}"
echo
echo "Corrected smooth + extended FITS:"
echo "  ${FINAL_FITS}"
echo
echo "Pointlike FITS:"
echo "  ${POINTLIKE_FITS}"
echo
echo "Final total FITS:"
echo "  ${TOTAL_FITS}"
echo "======================================================================"
