#!/bin/bash
# Merge sharded Dataset C outputs into one CSV (header kept once), for the
# downstream crop step (ROME_SIONNA_C_CSV).
#
# Usage:
#   slurm_datagen/c/concat_shards.sh "/data/ararat/rome_data/rome_c_constrained_shard*" \
#       /data/ararat/rome_data/rome_c_constrained/pretraining_dataset_patches.csv

set -euo pipefail
GLOB="${1:?usage: concat_shards.sh '<shard_dir_glob>' <out_csv>}"
OUT="${2:?usage: concat_shards.sh '<shard_dir_glob>' <out_csv>}"
mkdir -p "$(dirname "$OUT")"

first=1
: > "$OUT"
for dir in $GLOB; do
    f="$dir/pretraining_dataset_patches.csv"
    [ -f "$f" ] || { echo "skip (missing): $f"; continue; }
    if [ "$first" -eq 1 ]; then
        cat "$f" >> "$OUT"; first=0
    else
        tail -n +2 "$f" >> "$OUT"   # drop header on subsequent shards
    fi
    echo "merged: $f ($(($(wc -l < "$f")-1)) rows)"
done
echo "TOTAL rows in $OUT: $(($(wc -l < "$OUT")-1))"
