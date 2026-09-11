#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_EXECUTABLE:-python3}"

exec "${PYTHON}" "${SCRIPTS_DIR}/prepare_subhalo_components.py" "$@"
