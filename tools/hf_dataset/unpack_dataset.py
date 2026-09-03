"""Restore RF-Loc-Sim2Real datasets from Hugging Face crop shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

try:
    import zstandard as zstd
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency: zstandard. Install with `python -m pip install zstandard`.") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar
    tqdm = None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_tar_zst(shard: Path, target_dir: Path) -> None:
    target_resolved = target_dir.resolve()
    dctx = zstd.ZstdDecompressor()
    with shard.open("rb") as raw:
        with dctx.stream_reader(raw) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    destination = (target_dir / member.name).resolve()
                    if not str(destination).startswith(str(target_resolved)):
                        raise RuntimeError(f"Refusing to extract unsafe path {member.name!r} from {shard}")
                    tar.extract(member, target_dir)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="HF dataset export directory.")
    parser.add_argument("--output", required=True, type=Path, help="Restored dataset output directory.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional subset of datasets to restore.")
    parser.add_argument("--skip-checksums", action="store_true", help="Skip shard checksum validation.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    args = parser.parse_args()

    export_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    manifest_path = export_root / "dataset_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {output_root}. Use --overwrite to continue.")
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or list(manifest["datasets"].keys())
    for dataset in datasets:
        if dataset not in manifest["datasets"]:
            raise SystemExit(f"Dataset {dataset!r} not found in manifest.")
        print(f"Restoring {dataset}")
        src_dataset = export_root / dataset
        dst_dataset = output_root / dataset
        copy_tree(src_dataset / "splits", dst_dataset / "splits")
        copy_tree(src_dataset / "tabular", dst_dataset / "tabular")
        crops_dir = dst_dataset / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        shards = manifest["datasets"][dataset]["crops"]["shards"]
        iterator = shards
        if tqdm is not None:
            iterator = tqdm(shards, desc=f"extracting {dataset}", unit="shard")
        for shard_info in iterator:
            shard = src_dataset / "crops_shards" / shard_info["name"]
            if not shard.exists():
                raise SystemExit(f"Missing shard: {shard}")
            if not args.skip_checksums:
                actual = sha256_file(shard)
                if actual != shard_info["sha256"]:
                    raise SystemExit(f"Checksum mismatch for {shard}: {actual} != {shard_info['sha256']}")
            safe_extract_tar_zst(shard, crops_dir)

    print(f"Restored datasets to: {output_root}")


if __name__ == "__main__":
    main()
