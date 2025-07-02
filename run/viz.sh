#!/bin/sh

HYDRA_FULL_ERROR=1 \
python -m streamlit \
run \
../run.py \
-- \
--config-name=rome_visualize