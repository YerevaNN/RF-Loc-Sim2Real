#!/bin/sh

#SBATCH --partition=all
#SBATCH --job-name=rome_generate_data_var
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=/nfs/dgx/raid/iot/slurm/%x_%j.out
#SBATCH --error=/nfs/dgx/raid/iot/slurm/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --mem=20gb
#SBATCH --mail-type=ALL

HYDRA_FULL_ERROR=1 \
python ../run.py \
--config-name=rome_generate_data \
random_size_std=100 \
seed=42 \
out_dir=/nfs/dgx/raid/iot/data/rome_dark_roads_buildings_var
