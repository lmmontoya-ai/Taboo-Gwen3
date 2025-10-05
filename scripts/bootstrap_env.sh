#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a conda environment for the Taboo-Gwen3 project.
# Usage: ./scripts/bootstrap_env.sh [ENV_NAME]

ENV_NAME=${1:-taboo-gwen3}

>&2 echo "[Bootstrap] Creating conda environment: ${ENV_NAME}"
conda env remove --name "${ENV_NAME}" --yes >/dev/null 2>&1 || true
conda env create --name "${ENV_NAME}" --file environment.yml

ACTIVATE_FILE=".conda-activate-${ENV_NAME}.sh"
cat <<ACTIVATE > "${ACTIVATE_FILE}"
conda activate ${ENV_NAME}
if command -v uv >/dev/null 2>&1; then
  uv pip install --no-cache-dir --editable .
else
  python -m pip install --upgrade pip
  python -m pip install --no-cache-dir --editable .
fi
ACTIVATE

>&2 echo "[Bootstrap] Run 'source ${ACTIVATE_FILE}' to finish setup."
