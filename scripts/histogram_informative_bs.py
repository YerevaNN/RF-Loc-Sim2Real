"""
Per-sample histogram of informative-BS count for the B' prime training split.

A BS is "informative" if its RSSI is strictly greater than RSSI_FLOOR (default -90 dBm).
Bin 0 = sample with all BSs at -90 (fully uninformative).

Usage:
    python scripts/histogram_informative_bs.py \
        --splits-dir $ROME_SIONNA_SPLITS_PATH_B_PRIME \
        --main-dir   $ROME_SIONNA_OUT_DIR_B_PRIME
"""
import argparse
import json
import os
from collections import Counter

from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", required=True, help="Dir holding train.json (B' prime splits)")
    ap.add_argument("--main-dir", required=True, help="Dataset main dir holding info_dataSet.json and crop JSONs")
    ap.add_argument("--dataset-type", default="dataSet")
    ap.add_argument("--min-num-bs", type=int, default=3, help="Same as datamodule config")
    ap.add_argument("--hard-campaign", default="-1")
    ap.add_argument("--hard-train-ratio", type=float, default=0.0)
    ap.add_argument("--rssi-floor", type=float, default=-90.0)
    ap.add_argument("--feature", default="RSSI")
    args = ap.parse_args()

    train_json_path = os.path.join(args.splits_dir, "train.json")
    info_json_path = os.path.join(args.main_dir, f"info_{args.dataset_type}.json")

    with open(train_json_path) as f:
        split_info = json.load(f)
    with open(info_json_path) as f:
        info_json = json.load(f)

    crop_paths = []
    for campaign_id, ues in split_info.items():
        if campaign_id == args.hard_campaign:
            keep_n = int(len(ues) * args.hard_train_ratio)
            ues = dict(list(ues.items())[:keep_n])  # deterministic subset (no random seed)
        for ueid, samples in ues.items():
            for crop_id in samples.keys():
                bs_count = info_json[campaign_id][ueid][crop_id]
                if bs_count >= args.min_num_bs:
                    crop_paths.append(os.path.join(args.main_dir, campaign_id, ueid, f"{crop_id}.json"))

    print(f"Total training samples after filters: {len(crop_paths)}")

    histogram = Counter()
    total_bs_pairs = 0
    informative_bs_pairs = 0
    samples_all_uninformative = 0

    for cp in tqdm(crop_paths, desc="Scanning crops"):
        with open(cp) as f:
            info = json.load(f)
        bss = info["BaseStations"]
        n_informative = 0
        for bs in bss:
            val = bs["measurements"].get(args.feature, args.rssi_floor)
            if val > args.rssi_floor:
                n_informative += 1
        histogram[n_informative] += 1
        total_bs_pairs += len(bss)
        informative_bs_pairs += n_informative
        if n_informative == 0:
            samples_all_uninformative += 1

    print()
    print(f"Samples: {len(crop_paths)}")
    print(f"  All-uninformative ({args.feature} <= {args.rssi_floor} for every BS): "
          f"{samples_all_uninformative} ({samples_all_uninformative / len(crop_paths):.1%})")
    print(f"Total (UE,BS) pairs: {total_bs_pairs}")
    print(f"  Informative pairs: {informative_bs_pairs} ({informative_bs_pairs / total_bs_pairs:.1%})")
    print(f"  Uninformative pairs: {total_bs_pairs - informative_bs_pairs} "
          f"({(total_bs_pairs - informative_bs_pairs) / total_bs_pairs:.1%})")

    print()
    print("Histogram (# informative BSs per training sample):")
    print("  count_informative_bs   n_samples   bar")
    max_count = max(histogram.values())
    bar_w = 50
    for k in sorted(histogram):
        n = histogram[k]
        bar = "#" * max(1, int(bar_w * n / max_count))
        print(f"  {k:>20}   {n:>9}   {bar}")


if __name__ == "__main__":
    main()
