#!/bin/sh

HYDRA_FULL_ERROR=1 \
python ../../run.py \
--config-name=inference \
split=test \
datamodule=rome_data \
algorithm=rome_transformer_unet \
network=vit_pp \
datamodule.batch_size=2 \
datamodule.num_workers=12 \
datamodule.for_viz=True \
gpu=0 \
rome_idx=[0]