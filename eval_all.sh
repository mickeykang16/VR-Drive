#!/usr/bin/env bash

set -eo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

CKPT="work_dirs/detach_no_grad_cycle_fix_resume_reproduce/iter_58000.pth"
GPUS=4

# Directory and base name for logs based on CKPT path
LOG_DIR="$(dirname "$CKPT")"
CKPT_BASE="$(basename "$CKPT" .pth)"

CONFIGS=(
  "projects/configs/diffusiondrive_configs/ours_original.py"
  "projects/configs/diffusiondrive_configs/ours_novel_view_depth1.py"
  "projects/configs/diffusiondrive_configs/ours_novel_view_height-0.7.py"
  "projects/configs/diffusiondrive_configs/ours_novel_view_height1.py"
  "projects/configs/diffusiondrive_configs/ours_novel_view_pitch_-10.py"
  "projects/configs/diffusiondrive_configs/ours_novel_view_pitch_5.py"
)

for CFG in "${CONFIGS[@]}"; do
  CFG_BASE="$(basename "$CFG" .py)"
  LOG_FILE="${LOG_DIR}/${CKPT_BASE}_${CFG_BASE}.txt"

  echo "========================================"
  echo "Running evaluation for: $CFG"
  echo "Logging to: $LOG_FILE"
  echo "========================================"

  bash ./tools/dist_test.sh "$CFG" "$CKPT" $GPUS --deterministic --eval bbox 2>&1 | tee "$LOG_FILE"
done
