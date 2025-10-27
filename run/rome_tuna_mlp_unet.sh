#!/bin/sh

#SBATCH --partition=all
#SBATCH --job-name=rome_c_mlp_unet
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=/nfs/dgx/raid/iot/slurm/%x_%j.out
#SBATCH --error=/nfs/dgx/raid/iot/slurm/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=160gb
#SBATCH --mail-type=ALL

HYDRA_FULL_ERROR=1 \
python ../run.py \
trainer.devices=[0] \
network=mlp_unet \
datamodule=rome_data \
datamodule.batch_size=31 \
datamodule.num_workers=12 \
datamodule.max_num_bs=3 \
trainer.max_epochs=5 \
scheduler=wsd_scheduler \
scheduler.warmup_epochs=0 \
scheduler.decay_epochs=5 \
scheduler.stable_lr=2e-4 \
scheduler.interval=step \
ckpt_path=/nfs/dgx/raid/iot/outputs/2025-10-08_11-29-56.432173/checkpoints/hard/last.ckpt
