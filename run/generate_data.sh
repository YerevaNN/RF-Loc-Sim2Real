#!/bin/sh

HYDRA_FULL_ERROR=1 \
python ../../run.py \
--config-name=rome_generate_data \
num_points=1000 \
num_workers=null \
out_dir=/nfs/dgx/raid/iot/data/rome_debug