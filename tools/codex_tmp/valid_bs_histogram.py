#!/usr/bin/env python3
"""Count valid-RSSI base stations per crop and save a histogram.

This script is read-only with respect to the dataset tree. It reads:
  - info_dataSet.json / info_dataset.json / another info_*.json file
  - per-crop JSON files under <dataset_root>/<campaign>/<ue>/<crop>.json

It counts crops whose total BS count in the info JSON is at least
--min-total-bs, then computes how many considered BS entries have RSSI above
--rssi-threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass(frozen=True)
class CropRecord:
    campaign_id: str
    ue_id: str
    crop_id: str
    total_bs_count: int


@dataclass(frozen=True)
class CropResult:
    valid_bs_count: int | None
    considered_bs_count: int | None
    total_bs_count: int
    error: str | None


def find_info_json(dataset_root: Path, explicit_info_path: str | None) -> Path:
    if explicit_info_path:
        info_path = Path(explicit_info_path).expanduser()
        if not info_path.exists():
            raise FileNotFoundError(f"Info JSON does not exist: {info_path}")
        return info_path

    preferred_names = [
        "info_dataSet.json",
        "info_dataset.json",
        "info_Dataset.json",
        "info_dataSet_interp.json",
        "info_dataset_interp.json",
    ]
    for name in preferred_names:
        candidate = dataset_root / name
        if candidate.exists():
            return candidate

    matches = sorted(dataset_root.glob("info_*.json"))
    if not matches:
        raise FileNotFoundError(f"No info_*.json file found under: {dataset_root}")
    if len(matches) > 1:
        match_list = "\n".join(str(path) for path in matches)
        raise ValueError(
            "Multiple info_*.json files found. Pass --info-json explicitly.\n"
            f"{match_list}"
        )
    return matches[0]


def infer_exclude_interpolated(info_path: Path, include_interpolated: bool) -> bool:
    if include_interpolated:
        return False
    name = info_path.name.lower()
    return "dataset" in name and "interp" not in name


def iter_crop_records(info_json: dict, min_total_bs: int) -> Iterable[CropRecord]:
    for campaign_id, ue_points in info_json.items():
        for ue_id, crops in ue_points.items():
            for crop_id, total_bs_count in crops.items():
                if int(total_bs_count) >= min_total_bs:
                    yield CropRecord(
                        campaign_id=str(campaign_id),
                        ue_id=str(ue_id),
                        crop_id=str(crop_id),
                        total_bs_count=int(total_bs_count),
                    )


def count_candidate_crops(info_json: dict, min_total_bs: int) -> tuple[int, int]:
    total_crops = 0
    candidate_crops = 0
    for ue_points in info_json.values():
        for crops in ue_points.values():
            total_crops += len(crops)
            candidate_crops += sum(int(count) >= min_total_bs for count in crops.values())
    return total_crops, candidate_crops


def analyze_crop(
    dataset_root: Path,
    record: CropRecord,
    rssi_threshold: float,
    exclude_interpolated: bool,
) -> CropResult:
    crop_path = dataset_root / record.campaign_id / record.ue_id / f"{record.crop_id}.json"
    try:
        with crop_path.open("r") as file:
            crop_info = json.load(file)
    except Exception as exc:
        return CropResult(
            valid_bs_count=None,
            considered_bs_count=None,
            total_bs_count=record.total_bs_count,
            error=f"{type(exc).__name__}: {crop_path}: {exc}",
        )

    valid_bs_count = 0
    considered_bs_count = 0
    for bs in crop_info.get("BaseStations", []):
        if exclude_interpolated and bs.get("interpolated", False):
            continue
        considered_bs_count += 1
        rssi = bs.get("measurements", {}).get("RSSI")
        try:
            if float(rssi) > rssi_threshold:
                valid_bs_count += 1
        except (TypeError, ValueError):
            pass

    return CropResult(
        valid_bs_count=valid_bs_count,
        considered_bs_count=considered_bs_count,
        total_bs_count=record.total_bs_count,
        error=None,
    )


def run_bounded_parallel(
    dataset_root: Path,
    records: Iterable[CropRecord],
    total: int,
    rssi_threshold: float,
    exclude_interpolated: bool,
    workers: int,
) -> tuple[Counter[int], dict[str, int], list[str]]:
    histogram: Counter[int] = Counter()
    summary_counts = Counter()
    errors: list[str] = []
    max_pending = max(1, workers * 8)
    iterator = iter(records)

    progress = tqdm(total=total, desc="Scanning crop JSONs") if tqdm else None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        exhausted = False

        while not exhausted and len(pending) < max_pending:
            try:
                record = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending.add(
                executor.submit(
                    analyze_crop,
                    dataset_root,
                    record,
                    rssi_threshold,
                    exclude_interpolated,
                )
            )

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                summary_counts["processed"] += 1
                if progress:
                    progress.update(1)

                if result.error is not None:
                    summary_counts["errors"] += 1
                    if len(errors) < 20:
                        errors.append(result.error)
                else:
                    histogram[int(result.valid_bs_count)] += 1
                    if result.considered_bs_count != result.total_bs_count:
                        summary_counts["info_count_mismatches"] += 1

                while not exhausted and len(pending) < max_pending:
                    try:
                        record = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(
                        executor.submit(
                            analyze_crop,
                            dataset_root,
                            record,
                            rssi_threshold,
                            exclude_interpolated,
                        )
                    )

    if progress:
        progress.close()
    return histogram, dict(summary_counts), errors


def save_csv(histogram: Counter[int], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["valid_bs_count", "num_crops"])
        for valid_bs_count in sorted(histogram):
            writer.writerow([valid_bs_count, histogram[valid_bs_count]])


def save_histogram_png(histogram: Counter[int], png_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    xs = sorted(histogram)
    ys = [histogram[x] for x in xs]

    fig_width = max(8, min(18, len(xs) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=160)
    ax.bar(xs, ys, width=0.85, color="#2f6f8f")
    ax.set_xlabel("Number of valid BS in crop (RSSI > threshold)")
    ax.set_ylabel("Number of crops")
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For crops with at least --min-total-bs BS according to info JSON, "
            "count how many BS entries have RSSI above --rssi-threshold."
        )
    )
    parser.add_argument(
        "dataset_root",
        help="Path containing info_dataSet.json and campaign folders, e.g. .../rome_b_prime_20260203/const",
    )
    parser.add_argument(
        "--info-json",
        default=None,
        help="Explicit info JSON path. If omitted, the script searches under dataset_root.",
    )
    parser.add_argument(
        "--out-dir",
        default="tools/codex_tmp/valid_bs_stats",
        help="Directory where CSV, summary JSON, and histogram PNG will be written.",
    )
    parser.add_argument(
        "--rssi-threshold",
        type=float,
        default=-140.0,
        help="A BS is valid when RSSI is strictly greater than this value.",
    )
    parser.add_argument(
        "--min-total-bs",
        type=int,
        default=3,
        help="Only analyze crops whose info JSON BS count is at least this value.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 1) * 2),
        help="Number of parallel file-reading workers.",
    )
    parser.add_argument(
        "--include-interpolated",
        action="store_true",
        help="Count interpolated BSs too. By default, info_dataSet.json excludes them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    info_path = find_info_json(dataset_root, args.info_json)
    exclude_interpolated = infer_exclude_interpolated(info_path, args.include_interpolated)

    print(f"Dataset root: {dataset_root}")
    print(f"Info JSON: {info_path}")
    print(f"RSSI valid condition: RSSI > {args.rssi_threshold}")
    print(f"Crop inclusion condition: info_count >= {args.min_total_bs}")
    print(f"Exclude interpolated BS: {exclude_interpolated}")

    with info_path.open("r") as file:
        info_json = json.load(file)

    total_crops, candidate_crops = count_candidate_crops(info_json, args.min_total_bs)
    print(f"Total crops in info JSON: {total_crops}")
    print(f"Candidate crops to scan: {candidate_crops}")

    histogram, run_counts, errors = run_bounded_parallel(
        dataset_root=dataset_root,
        records=iter_crop_records(info_json, args.min_total_bs),
        total=candidate_crops,
        rssi_threshold=args.rssi_threshold,
        exclude_interpolated=exclude_interpolated,
        workers=args.workers,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "valid_bs_histogram.csv"
    png_path = out_dir / "valid_bs_histogram.png"
    summary_path = out_dir / "valid_bs_summary.json"

    save_csv(histogram, csv_path)
    save_histogram_png(
        histogram,
        png_path,
        title=f"Valid BS per crop, info_count >= {args.min_total_bs}",
    )

    summary = {
        "dataset_root": str(dataset_root),
        "info_json": str(info_path),
        "rssi_valid_condition": f"RSSI > {args.rssi_threshold}",
        "min_total_bs": args.min_total_bs,
        "exclude_interpolated": exclude_interpolated,
        "total_crops_in_info_json": total_crops,
        "candidate_crops_scanned": candidate_crops,
        "histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "run_counts": run_counts,
        "first_errors": errors,
        "outputs": {
            "csv": str(csv_path),
            "png": str(png_path),
            "summary": str(summary_path),
        },
    }
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
