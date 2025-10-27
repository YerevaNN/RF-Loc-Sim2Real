#!/bin/sh

#SBATCH --partition=all
#SBATCH --job-name=rome_b_prime_mlp_unet
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
datamodule=rome_sionna_b_prime_data \
datamodule.batch_size=31 \
datamodule.num_workers=32 \
datamodule.crops_per_epoch=0.4 \
datamodule.max_num_bs=3 \
trainer.max_epochs=100 \
scheduler=wsd_scheduler \
scheduler.interval=step
