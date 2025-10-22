#!/bin/sh

HYDRA_FULL_ERROR=1 \
python ../run.py \
--config-name=dann_inference \
split=test \
domain=target \
datamodule=rome_dann_data \
algorithm=rome_dann \
network=vit_pp_dann \
datamodule.batch_size=2 \
datamodule.num_workers=12 \
datamodule.for_viz=True \
gpu=0 \
rome_idx=[0,1,2]


