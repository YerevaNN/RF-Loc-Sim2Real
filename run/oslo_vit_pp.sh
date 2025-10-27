#!/bin/sh

#SBATCH --partition=all
#SBATCH --job-name=oslo_vit_pp
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --output=/nfs/dgx/raid/iot/slurm/%x_%j.out
#SBATCH --error=/nfs/dgx/raid/iot/slurm/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --time=36:00:00
#SBATCH --mem=160gb
#SBATCH --mail-type=ALL

HYDRA_FULL_ERROR=1 \
python ../run.py \
trainer.devices=-1 \
datamodule=oslo_data \
datamodule.batch_size=31 \
datamodule.num_workers=64 \
trainer.max_epochs=100 \
scheduler=wsd_scheduler \
scheduler.interval=step \
strategy=ddp \
strategy.find_unused_parameters=True \
loggers.aim.experiment=ifsi
