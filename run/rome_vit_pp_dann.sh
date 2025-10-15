#!/bin/sh

HYDRA_FULL_ERROR=1 \
python ../run.py \
--config-name=train_dann \
trainer.devices=-1 \
datamodule=rome_dann_data \
datamodule.batch_size=8 \
datamodule.num_workers=12 \
datamodule.crops_per_epoch=1 \
trainer.max_epochs=100 \
scheduler=wsd_scheduler \
scheduler.warmup_epochs=10 \
scheduler.decay_epochs=10 \
scheduler.stable_lr=2e-4 \
scheduler.warmup_start_lr=0 \
scheduler.eta_min=0 \
scheduler.interval=step \
strategy=ddp \
strategy.find_unused_parameters=True
