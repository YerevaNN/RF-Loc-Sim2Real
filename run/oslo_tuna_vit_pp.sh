#!/bin/sh

HYDRA_FULL_ERROR=1 \
python ../run.py \
trainer.max_epochs=5 \
trainer.devices=[0] \
datamodule=oslo_data \
datamodule.batch_size=31 \
datamodule.num_workers=12 \
scheduler=wsd_scheduler \
scheduler.warmup_epochs=0 \
scheduler.decay_epochs=5 \
scheduler.stable_lr=2e-4 \
scheduler.interval=step \
loggers.aim.experiment=ifsi \
ckpt_path=/nfs/dgx/raid/iot/outputs/2025-07-25_09-52-22.347492/checkpoints/hard/last.ckpt
