#!/bin/bash
#SBATCH --job-name=activity_run
#SBATCH --output=logs/activity_run_%j.out
#SBATCH --error=logs/activity_run_%j.err
#SBATCH --partition=TEST

gpu='1'

python run_models.py \
  --dataset activity \
  --state def \
  --history 24000 \
  --patience 10 \
  --batch_size 32 \
  --lr 1e-3 \
  --patch_size 8000 \
  --stride 4000 \
  --nhead 1 \
  --tf_layer 1 \
  --nlayer 1 \
  --te_dim 10 \
  --node_dim 10 \
  --hid_dim 64 \
  --outlayer Linear \
  --gpu "$gpu"
