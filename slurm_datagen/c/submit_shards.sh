#!/bin/bash
# Submit Dataset C generation as N parallel SLURM shards (one GPU each).
# Each shard owns a disjoint patch range -> disjoint scene/OSM/road writes,
# so shards are safe to run concurrently on a shared SCENE_CACHE.
#
# Usage:
#   slurm_datagen/c/submit_shards.sh <variant> <nshards>
#   slurm_datagen/c/submit_shards.sh constrained 4
#   slurm_datagen/c/submit_shards.sh unconstrained 4   # run AFTER constrained finishes (reuses cache)
#
# After all shards finish, merge with concat_shards.sh (header kept once).

set -euo pipefail
VARIANT="${1:-constrained}"
NSHARDS="${2:-4}"
TOTAL="${TOTAL_PATCHES:-144}"
BASE_OUT="${BASE_OUT:-/data/ararat/rome_data/rome_c_${VARIANT}}"
HERE="$(cd "$(dirname "$0")" && pwd)"

CHUNK=$(( (TOTAL + NSHARDS - 1) / NSHARDS ))   # ceil
echo "Submitting $NSHARDS shards x ~$CHUNK patches  (variant=$VARIANT, total=$TOTAL)"
for ((s=0; s<NSHARDS; s++)); do
    START=$(( s * CHUNK ))
    END=$(( START + CHUNK )); if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi
    if [ "$START" -ge "$TOTAL" ]; then break; fi
    OUT="${BASE_OUT}_shard${s}"
    echo "  shard $s: patches [$START,$END) -> $OUT"
    sbatch --job-name="rome_c_${VARIANT}_s${s}" \
        --export=ALL,BS_VARIANT="$VARIANT",START_PATCH="$START",END_PATCH="$END",OUT_DIR="$OUT" \
        "$HERE/generate_c.sh"
done
echo "Watch:  squeue -u \$USER ; tail -f /home/amanukyan/slurm_logs/datagen/c/rome_c_*_*.log"
