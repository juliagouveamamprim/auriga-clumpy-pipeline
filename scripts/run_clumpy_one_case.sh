#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run one complete CLUMPY pipeline case
# ============================================================
#
# Usage:
#
#   bash run_clumpy_one_case.sh <repop_id> <scenario>
#
# Examples:
#
#   bash run_clumpy_one_case.sh 1 resilient
#   bash run_clumpy_one_case.sh 1 fragile
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

if [ "$#" -ne 2 ]; then
    echo "Usage:"
    echo "  bash $0 <repop_id> <scenario>"
    echo
    echo "Examples:"
    echo "  bash $0 1 resilient"
    echo "  bash $0 1 fragile"
    exit 1
fi

REPOP_ID="$1"
SCENARIO="$2"

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

# Set CLUMPY_EXECUTABLE to the CLUMPY binary or wrapper.
# The executable must produce the patched *.halo_rendered.log output.
CLUMPY="${CLUMPY_EXECUTABLE:-clumpy}"

if ! command -v "$CLUMPY" >/dev/null 2>&1; then
    echo "ERROR: CLUMPY executable not found:"
    echo "  $CLUMPY"
    echo
    echo "Set it with, for example:"
    echo "  export CLUMPY_EXECUTABLE=/path/to/clumpy_wrapper"
    exit 1
fi

CASE_DIR="${BASE_RUN_DIR}/${SCENARIO}"

RAW_PARAM="${CASE_DIR}/params/generated/${REPOP_TAG}_raw_params.txt"
CORRECTED_PARAM="${CASE_DIR}/params/generated/${REPOP_TAG}_corrected_params.txt"

RAW_CLUMPY_LOG="${CASE_DIR}/logs/raw_clumpy/${REPOP_TAG}_raw_clumpy.log"
CORRECTED_CLUMPY_LOG="${CASE_DIR}/logs/corrected_clumpy/${REPOP_TAG}_corrected_clumpy.log"

RAW_OUTPUT_DIR="${CASE_DIR}/outputs/raw_clumpy/${REPOP_TAG}"
CORRECTED_OUTPUT_DIR="${CASE_DIR}/outputs/corrected_clumpy/${REPOP_TAG}"

RAW_RENDERED_LOG="${RAW_OUTPUT_DIR}/annihil_gal2D_LOS0_0_FOV360x180_nside1024.halo_rendered.log"
CORRECTED_RENDERED_LOG="${CORRECTED_OUTPUT_DIR}/annihil_gal2D_LOS0_0_FOV360x180_nside1024.halo_rendered.log"

FINAL_FITS="${CORRECTED_OUTPUT_DIR}/annihil_gal2D_LOS0_0_FOV360x180_nside1024.fits"
TOTAL_OUTPUT_DIR="${CASE_DIR}/outputs/total/${REPOP_TAG}"
TOTAL_FITS="${TOTAL_OUTPUT_DIR}/auriga_total_nside1024.fits"

echo
echo "======================================================================"
echo "Starting CLUMPY pipeline one-case run"
echo "======================================================================"
echo "REPOP_ID:  ${REPOP_ID}"
echo "REPOP_TAG: ${REPOP_TAG}"
echo "SCENARIO:  ${SCENARIO}"
echo "Time:      $(date)"
echo "======================================================================"
echo

if [ -f "$FINAL_FITS" ] && [ -s "$FINAL_FITS" ]; then
    echo "ERROR: final corrected FITS already exists and is non-empty:"
    echo "  $FINAL_FITS"
    echo
    echo "Refusing to overwrite an apparently completed run."
    exit 1
fi

# ============================================================
# 1. HDF5 -> raw CLUMPY list without pointlikes
# ============================================================

echo
echo "----------------------------------------------------------------------"
echo "Step 1/7: prepare extended list and pointlike HEALPix map"
echo "----------------------------------------------------------------------"

python3 "${SCRIPTS_DIR}/prepare_subhalo_components.py" \
    "${REPOP_ID}" "${SCENARIO}"

RAW_LIST="${CASE_DIR}/lists/raw/${REPOP_TAG}_raw_nopointlike.txt"
POINTLIKE_FITS="${CASE_DIR}/pointlike/${REPOP_TAG}_pointlike_nside1024.fits"

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

python3 "${SCRIPTS_DIR}/generate_clumpy_params.py" \
    "${REPOP_ID}" "${SCENARIO}" raw

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

python3 "${SCRIPTS_DIR}/correct_rhos_from_clumpy_raw.py" \
    "${REPOP_ID}" "${SCENARIO}"

CORRECTED_LIST="${CASE_DIR}/lists/corrected/${REPOP_TAG}_rhocorr.txt"

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

python3 "${SCRIPTS_DIR}/generate_clumpy_params.py" \
    "${REPOP_ID}" "${SCENARIO}" corrected

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

python3 "${SCRIPTS_DIR}/combine_clumpy_pointlike.py" \
    "${REPOP_ID}" "${SCENARIO}"

if [ ! -f "$TOTAL_FITS" ] || [ ! -s "$TOTAL_FITS" ]; then
    echo "ERROR: total FITS was not created or is empty:"
    echo "  $TOTAL_FITS"
    exit 1
fi

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
