#!/usr/bin/env python3
"""Create a Hugging Face upload-ready shard export for RF-Loc-Sim2Real data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import zstandard as zstd
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "Missing dependency: zstandard. Install with "
        "`python -m pip install -r tools/hf_dataset/requirements-export.txt`."
    ) from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_DATASETS = ("oslo_a", "rome_a", "rome_bp", "rome_c")
DEFAULT_SPLITS = (
    "train.json",
    "hard_val.json",
    "hard_test.json",
    "medium_val.json",
    "medium_test.json",
    "easy_val.json",
    "easy_test.json",
)


@dataclass
class FileRecord:
    path: Path
    relpath: str
    size: int


def parse_size(value: str) -> int:
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    text = value.strip().lower()
    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterable[FileRecord]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        current = Path(dirpath)
        for filename in filenames:
            path = current / filename
            relpath = path.relative_to(root).as_posix()
            yield FileRecord(path=path, relpath=relpath, size=path.stat().st_size)


def copy_tree_contents(src: Path, dst: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    files = []
    total_size = 0
    if not src.exists():
        return {"exists": False, "files": [], "file_count": 0, "total_size": 0}

    for record in iter_files(src):
        target = dst / record.relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.path, target)
        files.append({"path": record.relpath, "size": record.size, "sha256": sha256_file(target)})
        total_size += record.size
    return {"exists": True, "files": files, "file_count": len(files), "total_size": total_size}


def add_file_to_tar(tar: tarfile.TarFile, record: FileRecord) -> None:
    info = tar.gettarinfo(str(record.path), arcname=record.relpath)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with record.path.open("rb") as handle:
        tar.addfile(info, handle)


def write_zstd_tar(records: list[FileRecord], shard_path: Path, compression_level: int) -> None:
    cctx = zstd.ZstdCompressor(level=compression_level, threads=-1)
    with shard_path.open("wb") as raw:
        with cctx.stream_writer(raw) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                iterator = records
                if tqdm is not None:
                    iterator = tqdm(records, desc=f"writing {shard_path.name}", unit="file")
                for record in iterator:
                    add_file_to_tar(tar, record)


def make_shards(crops_root: Path, shard_dir: Path, max_shard_bytes: int, compression_level: int) -> list[dict]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    records = iter_files(crops_root)
    shards = []
    current: list[FileRecord] = []
    current_size = 0
    shard_idx = 0
    crop_file_count = 0
    crop_uncompressed_size = 0

    def flush() -> None:
        nonlocal current, current_size, shard_idx
        if not current:
            return
        shard_name = f"crops-{shard_idx:05d}.tar.zst"
        shard_path = shard_dir / shard_name
        write_zstd_tar(current, shard_path, compression_level)
        shards.append(
            {
                "name": shard_name,
                "file_count": len(current),
                "uncompressed_size": current_size,
                "compressed_size": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
            }
        )
        current = []
        current_size = 0
        shard_idx += 1

    for record in records:
        if current and current_size + record.size > max_shard_bytes:
            flush()
        current.append(record)
        current_size += record.size
        crop_file_count += 1
        crop_uncompressed_size += record.size
    flush()

    return [
        {
            "shard_count": len(shards),
            "file_count": crop_file_count,
            "uncompressed_size": crop_uncompressed_size,
            "shards": shards,
        }
    ][0]


def load_split_stats(splits_dir: Path) -> dict:
    stats = {}
    for split_name in DEFAULT_SPLITS:
        path = splits_dir / split_name
        if not path.exists():
            continue
        with path.open("r") as handle:
            data = json.load(handle)
        stats[split_name] = {
            "campaigns": len(data),
            "ues": sum(len(ues) for ues in data.values()),
            "sample_refs": sum(len(samples) for ues in data.values() for samples in ues.values()),
        }
    return stats


def write_dataset_card(output: Path, datasets: list[str], repo_id_hint: str | None) -> None:
    repo_line = f"`{repo_id_hint}`" if repo_id_hint else "this Hugging Face dataset repository"
    content = f"""---
license: other
pretty_name: RF-Loc-Sim2Real Crops
task_categories:
- image-to-image
- tabular-regression
---

# RF-Loc-Sim2Real Dataset Export

This dataset contains RF-Loc-Sim2Real crop data packaged as compressed crop shards.
It preserves the original project data layout after extraction.

Datasets included:

{chr(10).join(f'- `{name}`' for name in datasets)}

The crop files are stored in `crops_shards/*.tar.zst` to avoid uploading millions of
small files directly to Hugging Face. Split JSON files and tabular source files are
kept as normal files.

## Restore

Install the lightweight restore dependencies:

```bash
python -m pip install -r scripts/requirements.txt
```

Download or clone {repo_line}, then unpack:

```bash
python scripts/unpack_dataset.py \\
  --input . \\
  --output /path/to/restored_rf_loc_data
```

Verify the restored project-compatible layout:

```bash
python scripts/verify_rf_loc_format.py \\
  --root /path/to/restored_rf_loc_data
```

For RF-Loc-Sim2Real training configs, set dataset main paths to
`/path/to/restored_rf_loc_data/<dataset>/crops` and split paths to
`/path/to/restored_rf_loc_data/<dataset>/splits`.
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def copy_consumer_scripts(output: Path, script_dir: Path) -> None:
    target = output / "scripts"
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("unpack_dataset.py", "verify_rf_loc_format.py"):
        shutil.copy2(script_dir / filename, target / filename)
    (target / "requirements.txt").write_text("tqdm>=4.65.0\nzstandard>=0.22.0\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source dgx_backup directory.")
    parser.add_argument("--output", required=True, type=Path, help="Upload-ready export directory.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Dataset names to export.")
    parser.add_argument("--max-shard-size", default="20GB", help="Approximate max uncompressed bytes per crop shard.")
    parser.add_argument("--compression-level", type=int, default=10, help="Zstandard compression level.")
    parser.add_argument("--repo-id-hint", default=None, help="Optional repo id to mention in generated README.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory before writing.")
    args = parser.parse_args()

    source_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent

    if not source_root.exists():
        raise SystemExit(f"Input directory does not exist: {source_root}")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_root}. Use --overwrite to replace it.")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "rf-loc-sim2real-hf-sharded-v1",
        "created_at_unix": int(time.time()),
        "source_root": str(source_root),
        "datasets": {},
        "restore_layout": "<output>/<dataset>/{crops,splits,tabular}",
        "notes": [
            "Crop shards contain paths relative to each dataset's crops directory.",
            "After unpacking, RF-Loc-Sim2Real dataset_main_paths should point to <dataset>/crops.",
        ],
    }

    max_shard_bytes = parse_size(args.max_shard_size)
    for dataset in args.datasets:
        src_dataset = source_root / dataset
        if not src_dataset.exists():
            raise SystemExit(f"Dataset directory not found: {src_dataset}")

        dst_dataset = output_root / dataset
        dst_dataset.mkdir(parents=True, exist_ok=True)
        print(f"Copying metadata for {dataset}")
        splits_info = copy_tree_contents(src_dataset / "splits", dst_dataset / "splits")
        tabular_info = copy_tree_contents(src_dataset / "tabular", dst_dataset / "tabular")

        print(f"Creating crop shards for {dataset}")
        crops_info = make_shards(
            src_dataset / "crops",
            dst_dataset / "crops_shards",
            max_shard_bytes=max_shard_bytes,
            compression_level=args.compression_level,
        )

        manifest["datasets"][dataset] = {
            "splits": splits_info,
            "tabular": tabular_info,
            "crops": crops_info,
            "split_stats": load_split_stats(src_dataset / "splits"),
        }

    copy_consumer_scripts(output_root, script_dir)
    write_dataset_card(output_root, args.datasets, args.repo_id_hint)
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote export manifest: {manifest_path}")
    print(f"Upload-ready dataset export: {output_root}")


if __name__ == "__main__":
    main()
