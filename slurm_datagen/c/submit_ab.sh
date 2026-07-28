#!/bin/bash
# Submit BOTH variants as N shards each, plus an auto-merge job per variant, in
# one shot. Fire once -> two clean per-variant datasets at separate /mnt/weka
# paths, no manual merging.
#
# Dependencies (SLURM afterok):
#   * The N constrained shards run immediately, in parallel (disjoint patch
#     ranges -> disjoint scene/OSM writes, safe concurrently).
#   * Each unconstrained shard s waits for constrained shard s (same patches),
#     then reuses the scenes it built.
#   * A per-variant MERGE job waits for all that variant's shards, then
#     concatenates shard CSVs into one <variant>/pretraining_dataset_patches.csv.
#
# Usage:
#   slurm_datagen/c/submit_ab.sh 8                 # 8 shards/variant (defaults)
#   N_FAKE=100000 slurm_datagen/c/submit_ab.sh 8   # 100k UEs/patch (large run)
#
# Layout (override BASE to relocate):
#   <BASE>/constrained/shard{0..N-1}/  + <BASE>/constrained/pretraining_dataset_patches.csv
#   <BASE>/unconstrained/shard{0..N-1}/+ <BASE>/unconstrained/pretraining_dataset_patches.csv
#   <BASE>/scene_cache/   (shared)

set -euo pipefail
NSHARDS="${1:-8}"
TOTAL="${TOTAL_PATCHES:-144}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE:-/mnt/weka/amanukyan/rome_c}"
CON_DIR="$BASE/constrained"
UNC_DIR="$BASE/unconstrained"
export SCENE_CACHE="${SCENE_CACHE:-$BASE/scene_cache}"

CHUNK=$(( (TOTAL + NSHARDS - 1) / NSHARDS ))
echo "Submitting $NSHARDS shards/variant (~$CHUNK patches each) -> $BASE"

con_ids=""; unc_ids=""
for ((s=0; s<NSHARDS; s++)); do
    START=$(( s * CHUNK ))
    END=$(( START + CHUNK )); if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi
    if [ "$START" -ge "$TOTAL" ]; then break; fi

    cid=$(sbatch --parsable --job-name="rc_con_s${s}" \
        --export=ALL,BS_VARIANT=constrained,START_PATCH="$START",END_PATCH="$END",OUT_DIR="$CON_DIR/shard${s}" \
        "$HERE/generate_c.sh")
    uid=$(sbatch --parsable --dependency=afterok:"$cid" --job-name="rc_unc_s${s}" \
        --export=ALL,BS_VARIANT=unconstrained,START_PATCH="$START",END_PATCH="$END",OUT_DIR="$UNC_DIR/shard${s}" \
        "$HERE/generate_c.sh")

    con_ids="${con_ids:+$con_ids:}$cid"
    unc_ids="${unc_ids:+$unc_ids:}$uid"
    echo "  shard $s [$START,$END):  con=$cid -> unc=$uid"
done

# Per-variant auto-merge (CPU-only, waits for all that variant's shards).
mc=$(sbatch --parsable --job-name="rc_con_merge" --dependency=afterok:"$con_ids" \
     --partition=research --time=06:00:00 --mem=16G --cpus-per-task=2 \
     --output=/home/amanukyan/slurm_logs/datagen/c/merge_con_%j.log \
     --wrap="bash '$HERE/concat_shards.sh' '$CON_DIR/shard*' '$CON_DIR/pretraining_dataset_patches.csv'")
mu=$(sbatch --parsable --job-name="rc_unc_merge" --dependency=afterok:"$unc_ids" \
     --partition=research --time=06:00:00 --mem=16G --cpus-per-task=2 \
     --output=/home/amanukyan/slurm_logs/datagen/c/merge_unc_%j.log \
     --wrap="bash '$HERE/concat_shards.sh' '$UNC_DIR/shard*' '$UNC_DIR/pretraining_dataset_patches.csv'")

echo
echo "merge jobs:  constrained=$mc  unconstrained=$mu"
echo "Final CSVs (auto-created):"
echo "  $CON_DIR/pretraining_dataset_patches.csv"
echo "  $UNC_DIR/pretraining_dataset_patches.csv"
echo "Watch:  squeue -u \$USER"
