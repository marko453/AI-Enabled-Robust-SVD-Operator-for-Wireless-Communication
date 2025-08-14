#!/usr/bin/env bash
set -euo pipefail
python model.py \
  --mode prune_ft \
  --data_dir ./data2 \
  --src_ckpt ./ckpts/preprune.pth \
  --keep_klen 28 \
  --keep_hidden 28 \
  --ft_epochs 60 \
  --ft_lr 2e-4
