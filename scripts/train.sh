#!/usr/bin/env bash
set -euo pipefail
python model.py \
  --mode train \
  --data_dir ./data2 \
  --out_dir ./submissions
