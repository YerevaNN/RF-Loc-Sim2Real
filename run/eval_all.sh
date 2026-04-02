#!/bin/bash
cd /home/amanukyan/RF-Loc-Sim2Real

# 1. B'' unconst 8GPU 300ep on B'' test
echo "=== B'' unconst 8GPU 300ep on B'' test ==="
ROME_OUT_DIR=/mnt/weka/asaribekyan/rome_b_prime_20260203/unconst ROME_SPLITS_PATH=/mnt/weka/amanukyan/iot/outputs/2026-03-27_07-00-28 OUTPUT_DIR=/mnt/weka/amanukyan/iot/outputs HYDRA_FULL_ERROR=1 python run.py --config-name=evaluation prediction_path=/mnt/weka/amanukyan/iot/preds/b_pp_unconst_8gpu_300ep/2026-03-28_01-45-23.086357

# 2. B'' const 8GPU 300ep on B'' test
echo "=== B'' const 8GPU 300ep on B'' test ==="
ROME_OUT_DIR=/mnt/weka/asaribekyan/rome_b_prime_20260203/const ROME_SPLITS_PATH=/mnt/weka/amanukyan/iot/outputs/2026-03-26_01-16-09 OUTPUT_DIR=/mnt/weka/amanukyan/iot/outputs HYDRA_FULL_ERROR=1 python run.py --config-name=evaluation prediction_path=/mnt/weka/amanukyan/iot/preds/b_pp_const_8gpu_300ep/2026-03-28_02-05-21.047487

# 3. Unconst 300ep zero-shot on A
echo "=== Unconst 300ep zero-shot on A ==="
ROME_OUT_DIR=/mnt/weka/asaribekyan/rome_data/rome_a_1000x1000_257crops ROME_SPLITS_PATH=/mnt/weka/amanukyan/iot/outputs/2026-03-18_01-25-07 OUTPUT_DIR=/mnt/weka/amanukyan/iot/outputs HYDRA_FULL_ERROR=1 python run.py --config-name=evaluation prediction_path=/mnt/weka/amanukyan/iot/preds/b_pp_unconst_300ep_on_A/2026-03-28_01-45-23.086357

# 4. Const 300ep zero-shot on A
echo "=== Const 300ep zero-shot on A ==="
ROME_OUT_DIR=/mnt/weka/asaribekyan/rome_data/rome_a_1000x1000_257crops ROME_SPLITS_PATH=/mnt/weka/amanukyan/iot/outputs/2026-03-18_01-25-07 OUTPUT_DIR=/mnt/weka/amanukyan/iot/outputs HYDRA_FULL_ERROR=1 python run.py --config-name=evaluation prediction_path=/mnt/weka/amanukyan/iot/preds/b_pp_const_300ep_on_A/2026-03-28_02-05-21.047487

# 5. Unconst 100ep zero-shot on A
echo "=== Unconst 100ep zero-shot on A ==="
ROME_OUT_DIR=/mnt/weka/asaribekyan/rome_data/rome_a_1000x1000_257crops ROME_SPLITS_PATH=/mnt/weka/amanukyan/iot/outputs/2026-03-18_01-25-07 OUTPUT_DIR=/mnt/weka/amanukyan/iot/outputs HYDRA_FULL_ERROR=1 python run.py --config-name=evaluation prediction_path=/mnt/weka/amanukyan/iot/preds/b_pp_unconst_100ep_on_A/2026-03-27_13-19-45.933355
