"""Verify RF-Loc-Sim2Real restored data or an HF sharded export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SPLITS = (
    "train.json",
    "hard_val.json",
    "hard_test.json",
    "medium_val.json",
    "medium_test.json",
    "easy_val.json",
    "easy_test.json",
)


def iter_datasets(root: Path, selected: list[str] | None) -> list[str]:
    if selected:
        return selected
    return sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))


def verify_split_refs(dataset_root: Path, sample_limit: int | None) -> dict:
    crops = dataset_root / "crops"
    splits = dataset_root / "splits"
    if not crops.exists():
        raise RuntimeError(f"Missing crops directory: {crops}")
    if not splits.exists():
        raise RuntimeError(f"Missing splits directory: {splits}")
    if not (crops / "info_dataSet.json").exists():
        raise RuntimeError(f"Missing required crop metadata: {crops / 'info_dataSet.json'}")

    stats = {}
    checked = 0
    for split_name in DEFAULT_SPLITS:
        split_path = splits / split_name
        if not split_path.exists():
            raise RuntimeError(f"Missing split file: {split_path}")
        data = json.loads(split_path.read_text(encoding="utf-8"))
        refs = 0
        for campaign, ues in data.items():
            for ue, samples in ues.items():
                for crop_id in samples:
                    refs += 1
                    if sample_limit is None or checked < sample_limit:
                        json_path = crops / campaign / ue / f"{crop_id}.json"
                        npz_path = crops / campaign / ue / f"{crop_id}.npz"
                        if not json_path.exists():
                            raise RuntimeError(f"Missing crop JSON referenced by split: {json_path}")
                        if not npz_path.exists():
                            raise RuntimeError(f"Missing crop NPZ referenced by split: {npz_path}")
                        checked += 1
        stats[split_name] = {
            "campaigns": len(data),
            "ues": sum(len(ues) for ues in data.values()),
            "sample_refs": refs,
        }
    stats["checked_crop_pairs"] = checked
    return stats


def verify_export_dataset(dataset_root: Path, manifest_dataset: dict) -> dict:
    if not (dataset_root / "splits").exists():
        raise RuntimeError(f"Missing export splits directory: {dataset_root / 'splits'}")
    if not (dataset_root / "tabular").exists():
        raise RuntimeError(f"Missing export tabular directory: {dataset_root / 'tabular'}")
    shard_dir = dataset_root / "crops_shards"
    if not shard_dir.exists():
        raise RuntimeError(f"Missing crop shard directory: {shard_dir}")

    missing = []
    for shard in manifest_dataset["crops"]["shards"]:
        path = shard_dir / shard["name"]
        if not path.exists():
            missing.append(str(path))
            continue
        if path.stat().st_size != shard["compressed_size"]:
            raise RuntimeError(f"Shard size mismatch: {path}")
    if missing:
        raise RuntimeError("Missing shards:\n" + "\n".join(missing[:20]))

    crops = manifest_dataset["crops"]
    report = {
        "shards": len(crops["shards"]),
        "split_stats": manifest_dataset.get("split_stats", {}),
    }
    if "file_count" in crops:
        report["crop_files_in_shards"] = crops["file_count"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Restored root or HF export root.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to verify.")
    parser.add_argument("--export", action="store_true", help="Verify the sharded HF export instead of restored data.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1000,
        help="Max split-referenced crop pairs to check per dataset. Use --full for all.",
    )
    parser.add_argument("--full", action="store_true", help="Check every split-referenced crop pair.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    manifest = None
    if args.export:
        manifest_path = root / "dataset_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing manifest for export verification: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = args.datasets or list(manifest["datasets"].keys())
    else:
        datasets = iter_datasets(root, args.datasets)

    report = {}
    for dataset in datasets:
        dataset_root = root / dataset
        if not dataset_root.exists():
            raise SystemExit(f"Dataset root does not exist: {dataset_root}")
        if args.export:
            if dataset not in manifest["datasets"]:
                raise SystemExit(f"Dataset {dataset!r} not present in manifest.")
            report[dataset] = verify_export_dataset(dataset_root, manifest["datasets"][dataset])
        else:
            report[dataset] = verify_split_refs(dataset_root, None if args.full else args.sample_limit)

    print(json.dumps(report, indent=2, sort_keys=True))
    print("Verification passed.")


if __name__ == "__main__":
    main()
