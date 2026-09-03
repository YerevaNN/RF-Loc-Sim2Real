"""Recompute a manifest for an existing HF dataset export."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional unless --inspect-shards is used
    zstd = None


DEFAULT_SPLITS = (
    "train.json",
    "hard_val.json",
    "hard_test.json",
    "medium_val.json",
    "medium_test.json",
    "easy_val.json",
    "easy_test.json",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entries(root: Path) -> dict:
    files = []
    total = 0
    if not root.exists():
        return {"exists": False, "files": [], "file_count": 0, "total_size": 0}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = path.stat().st_size
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": size,
                "sha256": sha256_file(path),
            }
        )
        total += size
    return {"exists": True, "files": files, "file_count": len(files), "total_size": total}


def load_split_stats(splits_dir: Path) -> dict:
    stats = {}
    for split_name in DEFAULT_SPLITS:
        path = splits_dir / split_name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        stats[split_name] = {
            "campaigns": len(data),
            "ues": sum(len(ues) for ues in data.values()),
            "sample_refs": sum(len(samples) for ues in data.values() for samples in ues.values()),
        }
    return stats


def inspect_tar_zst(shard: Path) -> tuple[int, int]:
    if zstd is None:
        raise SystemExit("Missing dependency: zstandard. Install with `python -m pip install zstandard`.")
    count = 0
    total = 0
    dctx = zstd.ZstdDecompressor()
    with shard.open("rb") as raw:
        with dctx.stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if member.isfile():
                        count += 1
                        total += member.size
    return count, total


def make_crop_shard_entries(dataset_root: Path, inspect_shards: bool) -> dict:
    shards = []
    file_count = 0
    uncompressed_size = 0
    for shard in sorted((dataset_root / "crops_shards").glob("*.tar.zst")):
        entry = {
            "name": shard.name,
            "compressed_size": shard.stat().st_size,
            "sha256": sha256_file(shard),
        }
        if inspect_shards:
            shard_files, shard_size = inspect_tar_zst(shard)
            entry["file_count"] = shard_files
            entry["uncompressed_size"] = shard_size
            file_count += shard_files
            uncompressed_size += shard_size
        shards.append(entry)

    crops = {"shard_count": len(shards), "shards": shards}
    if inspect_shards:
        crops["file_count"] = file_count
        crops["uncompressed_size"] = uncompressed_size
    return crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="HF dataset export root.")
    parser.add_argument("--output", type=Path, default=None, help="Manifest output path.")
    parser.add_argument(
        "--inspect-shards",
        action="store_true",
        help="Scan tar members to include crop file counts and uncompressed sizes.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    manifest = {
        "format": "rf-loc-sim2real-hf-sharded-v1",
        "created_at_unix": int(time.time()),
        "datasets": {},
        "restore_layout": "<output>/<dataset>/{crops,splits,tabular}",
    }

    for dataset_root in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "scripts"):
        manifest["datasets"][dataset_root.name] = {
            "splits": file_entries(dataset_root / "splits"),
            "tabular": file_entries(dataset_root / "tabular"),
            "crops": make_crop_shard_entries(dataset_root, args.inspect_shards),
            "split_stats": load_split_stats(dataset_root / "splits"),
        }

    output = args.output.expanduser().resolve() if args.output else root / "dataset_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {output}")


if __name__ == "__main__":
    main()
