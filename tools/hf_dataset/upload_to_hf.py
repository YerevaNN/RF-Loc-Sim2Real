"""Upload an RF-Loc-Sim2Real HF export directory to Hugging Face Datasets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install with "
        "`python -m pip install -r tools/hf_dataset/requirements-export.txt`."
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, e.g. org/name.")
    parser.add_argument("--local-dir", required=True, type=Path, help="Prepared hf_dataset_export directory.")
    parser.add_argument("--private", action="store_true", help="Create repo as private if it does not exist.")
    parser.add_argument("--revision", default="main", help="Target branch/revision.")
    parser.add_argument("--token", default=None, help="HF token. Defaults to HF_TOKEN or logged-in token.")
    parser.add_argument(
        "--no-hf-transfer",
        action="store_true",
        help="Do not set HF_HUB_ENABLE_HF_TRANSFER=1 for faster uploads.",
    )
    args = parser.parse_args()

    local_dir = args.local_dir.expanduser().resolve()
    if not local_dir.exists():
        raise SystemExit(f"Local export directory does not exist: {local_dir}")
    if not (local_dir / "dataset_manifest.json").exists():
        raise SystemExit(f"Missing dataset_manifest.json in {local_dir}")

    if not args.no_hf_transfer:
        try:
            __import__("hf_transfer")
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        except ImportError:
            print("hf_transfer is not installed; continuing with the standard Hugging Face upload path.")

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        revision=args.revision,
    )
    print(f"Uploaded dataset export to: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
