#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd "$SCRIPT_DIR/.." && pwd)

if [ -z "$ROME_SIONNA_SPLITS_PATH_C" ]; then
    ROME_SIONNA_SPLITS_PATH_C=/mnt/weka/asaribekyan/iot/outputs/2026-06-16_09-22-12
fi
export ROME_SIONNA_SPLITS_PATH_C

HYDRA_FULL_ERROR=1 \
python -m streamlit \
run \
"$REPO_ROOT/run.py" \
--server.address=localhost \
--server.port=8501 \
-- \
--config-name=rome_sionna_c_visualize
