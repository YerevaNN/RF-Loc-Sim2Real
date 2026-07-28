#!/bin/bash
#SBATCH --partition=research
#SBATCH --job-name=rome_c_datagen
#SBATCH --output=/home/amanukyan/slurm_logs/datagen/c/rome_c_%j.log
#SBATCH --error=/home/amanukyan/slurm_logs/datagen/c/rome_c_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=ALL

# =========================================================================
# Dataset C generation: fully-synthetic BS + UE on a city-scale patch grid.
#   * BS:  synthetic, variant=constrained (rooftop, roof-height + 3-6 m mast,
#          capped to the empirical [15,40] m band) or unconstrained (free).
#   * UE:  street strategy (on-road near BS, in-building rejected), like B''.
#   * Priors (altitude/power/azimuth) from configs/bs_patterns.json, fitted to
#     the real B'' const/unconst optimization results.
#
# Geometry (Phase 0): 2 km square patches, 12x12 = 144 patches.
#   24 km span: lat 24000/111000 = 0.2162 deg ; lon 24000/(111000*cos41.84) = 0.2904 deg
#   Centred on the Rome bbox centre (41.8425, 12.4775).
#
# Run (single job, all 144 patches, resumable via checkpoint.json):
#   sbatch slurm_datagen/c/generate_c.sh
# Override variant / output / shard range via --export, e.g.:
#   sbatch --export=ALL,BS_VARIANT=unconstrained,OUT_DIR=/path/uncon generate_c.sh
#   sbatch --export=ALL,START_PATCH=0,END_PATCH=72,OUT_DIR=/path/shardA generate_c.sh
# Shards MUST use distinct OUT_DIR; SCENE_CACHE / ROAD_CACHE can be shared.
# =========================================================================

# NOTE: do NOT enable `set -u` around conda activation — conda's own
# deactivate hooks (e.g. geotiff-deactivate.sh) reference unbound vars and
# would abort the job before Python starts. Enable strict mode AFTER activate,
# and without nounset.
source /home/amanukyan/miniconda3/etc/profile.d/conda.sh
conda activate rome
set -eo pipefail

# Per-job OptiX disk cache on node-local /tmp (avoids OPTIX_ERROR_DISK_CACHE
# collisions when jobs share a node — see B'' runner notes).
export OPTIX_CACHE_PATH="/tmp/optix_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$OPTIX_CACHE_PATH"
export TF_CPP_MIN_LOG_LEVEL=2

mkdir -p /home/amanukyan/slurm_logs/datagen/c

REPO=/home/amanukyan/RF-Loc-Sim2Real
# The simulation package uses `from src.X` / `from config.X`, so simulation/
# must be the import root (NOT the repo root, which has its own src/).
cd "$REPO/simulation"

# ----- overridable parameters -----
BS_VARIANT="${BS_VARIANT:-constrained}"
UE_STRATEGY="${UE_STRATEGY:-street}"
N_FAKE="${N_FAKE:-500}"
BS_PER_PATCH="${BS_PER_PATCH:-15}"
BATCH="${BATCH:-256}"
# UE_MAX_DIST_M="${UE_MAX_DIST_M:-600}"   # OLD: wide radius, sparser BS coverage
UE_MAX_DIST_M="${UE_MAX_DIST_M:-250}"   # match B'' street_d250 (UE within 250 m of a BS)
SIDEWALK_JITTER_M="${SIDEWALK_JITTER_M:-4}"
UE_HEIGHT_M="${UE_HEIGHT_M:-1.5}"
SEED="${SEED:-0}"
START_PATCH="${START_PATCH:-0}"
END_PATCH="${END_PATCH:-}"   # empty = all patches

# Each variant lands at its OWN /mnt/weka path; SCENE_CACHE is shared (content-
# addressed by bounds+grid, so constrained builds it and unconstrained reuses).
OUT_DIR="${OUT_DIR:-/mnt/weka/amanukyan/rome_c/$(printf '%s' "$BS_VARIANT")}"
SCENE_CACHE="${SCENE_CACHE:-/mnt/weka/amanukyan/rome_c/scene_cache}"
ROAD_CACHE="${ROAD_CACHE:-/mnt/weka/amanukyan/wair_d_iot/road_cache}"
PATTERNS_JSON="${PATTERNS_JSON:-$REPO/configs/bs_patterns.json}"

# 2 km, 12x12 over the 24 km Rome bbox (overridable for a smoke run)
LAT_MIN="${LAT_MIN:-41.7344}" ; LAT_MAX="${LAT_MAX:-41.9506}"
LON_MIN="${LON_MIN:-12.3323}" ; LON_MAX="${LON_MAX:-12.6227}"
GRID_ROWS="${GRID_ROWS:-12}" ; GRID_COLS="${GRID_COLS:-12}"

mkdir -p "$OUT_DIR" "$SCENE_CACHE" "$ROAD_CACHE"

END_ARG=()
if [ -n "$END_PATCH" ]; then END_ARG=( --end_patch "$END_PATCH" ); fi
# Optional: raise the per-call path budget when increasing BATCH (else paths/rx starve).
MAX_PATHS="${MAX_PATHS:-}"
PATHS_ARG=()
if [ -n "$MAX_PATHS" ]; then PATHS_ARG=( --max_num_paths "$MAX_PATHS" ); fi

# Solver provenance (depth/specular set in simulation/config/parameters.py).
echo "Dataset C generation: variant=$BS_VARIANT strategy=$UE_STRATEGY out=$OUT_DIR"
echo "Solver: freq=1.2GHz max_depth=5 specular=True diffuse=True refraction=True los=True samples_per_src=1e6 max_num_paths_per_src=${MAX_PATHS:-1e4(default)} (B'' floor-fix config)"
echo "UE: n=$N_FAKE strategy=$UE_STRATEGY max_dist=${UE_MAX_DIST_M}m height=${UE_HEIGHT_M}m | BS: $BS_PER_PATCH/patch variant=$BS_VARIANT | BATCH=$BATCH"
python -u -m src.pretraining_data_generator_patches \
    --out_dir "$OUT_DIR" \
    --n_fake "$N_FAKE" \
    --batch "$BATCH" \
    "${PATHS_ARG[@]}" \
    --lat_min "$LAT_MIN" --lat_max "$LAT_MAX" \
    --lon_min "$LON_MIN" --lon_max "$LON_MAX" \
    --grid_rows "$GRID_ROWS" --grid_cols "$GRID_COLS" \
    --bs_per_patch "$BS_PER_PATCH" \
    --bs_variant "$BS_VARIANT" \
    --ue_strategy "$UE_STRATEGY" \
    --ue_max_dist_m "$UE_MAX_DIST_M" \
    --sidewalk_jitter_m "$SIDEWALK_JITTER_M" \
    --ue_height_m "$UE_HEIGHT_M" \
    --scene_cache "$SCENE_CACHE" \
    --road_cache_dir "$ROAD_CACHE" \
    --patterns_json "$PATTERNS_JSON" \
    --seed "$SEED" \
    --start_patch "$START_PATCH" \
    "${END_ARG[@]}"

echo "Done. Output CSV: $OUT_DIR/pretraining_dataset_patches.csv"
