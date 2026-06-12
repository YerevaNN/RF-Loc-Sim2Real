# Hugging Face Dataset Export Tools

These scripts package the RF-Loc-Sim2Real crop datasets for Hugging Face Datasets while preserving the project-native format after extraction.

Maintainer flow:

```bash
python tools/hf_dataset/pack_dataset.py \
  --input /mnt/weka/asaribekyan/iot/data/dgx_backup \
  --output /mnt/weka/asaribekyan/iot/data/hf_dataset_export

python tools/hf_dataset/verify_rf_loc_format.py \
  --root /mnt/weka/asaribekyan/iot/data/hf_dataset_export \
  --export

python tools/hf_dataset/upload_to_hf.py \
  --repo-id <org-or-user>/<dataset-name> \
  --local-dir /mnt/weka/asaribekyan/iot/data/hf_dataset_export \
  --private
```

The packer writes consumer-facing restore files into the export directory automatically:

```text
README.md
dataset_manifest.json
scripts/unpack_dataset.py
scripts/verify_rf_loc_format.py
scripts/requirements.txt
```

The generated dataset should be uploaded from the export directory, not from the original `dgx_backup` tree.
