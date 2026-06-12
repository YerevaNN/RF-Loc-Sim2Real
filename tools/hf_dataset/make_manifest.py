#!/usr/bin/env python3
"""Recompute a manifest for an existing HF dataset export."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
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
        files.append({"path": path.relative_to(root).as_posix(), "size": size, "sha256": sha256_file(path)})
        total += size
    return {"exists": True, "files": files, "file_count": len(files), "total_size": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="HF dataset export root.")
    parser.add_argument("--output", type=Path, default=None, help="Manifest output path.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    manifest = {
        "format": "rf-loc-sim2real-hf-sharded-v1",
        "created_at_unix": int(time.time()),
        "datasets": {},
        "restore_layout": "<output>/<dataset>/{crops,splits,tabular}",
    }
    for dataset_root in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "scripts"):
        shards = []
        for shard in sorted((dataset_root / "crops_shards").glob("*.tar.zst")):
            shards.append(
                {
                    "name": shard.name,
                    "compressed_size": shard.stat().st_size,
                    "sha256": sha256_file(shard),
                }
            )
        manifest["datasets"][dataset_root.name] = {
            "splits": file_entries(dataset_root / "splits"),
            "tabular": file_entries(dataset_root / "tabular"),
            "crops": {"shard_count": len(shards), "shards": shards},
        }

    output = args.output.expanduser().resolve() if args.output else root / "dataset_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {output}")


if __name__ == "__main__":
    main()
