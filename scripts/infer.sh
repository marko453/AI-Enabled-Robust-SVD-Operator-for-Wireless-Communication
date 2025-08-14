#!/usr/bin/env bash
set -euo pipefail
python model.py \
  --mode infer \
  --data_dir ./data2 \
  --ckpt ./ckpts/best.pth \
  --out_dir ./submissions/round2
