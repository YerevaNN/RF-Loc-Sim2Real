#!/bin/sh

HYDRA_FULL_ERROR=1 \
python ../../run.py \
trainer.max_epochs=9999 \
trainer.devices=[1] \
datamodule.batch_size=10 \
datamodule.num_workers=12 \
optimizer.lr=3e-4 \
loggers.aim.experiment=ifsi