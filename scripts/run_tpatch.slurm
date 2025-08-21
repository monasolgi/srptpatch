#!/bin/bash
#SBATCH --job-name=physionet_run
#SBATCH --output=logs/physionet_run_%j.out
#SBATCH --error=logs/physionet_run_%j.err
#SBATCH --partition=TEST              # Specify the correct partition (not stud)


# Fix: Add lib to PYTHONPATH




gpu='1'



python run_models.py \
--dataset physionet \
--state 'def' \
--history 24 \
--patience 10 \
--batch_size 32 \
--lr 1e-3 \
--patch_size 8 \
--stride 8 \
--nhead 1 \
--tf_layer 1 \
--nlayer 1 \
--te_dim 10 \
--node_dim 10 \
--hid_dim 64 \
--outlayer Linear \
--gpu $gpu
#--seed $seed \