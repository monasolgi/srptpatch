### tPatchGNN ###
patience=10 #Early stopping patience
gpu=0

for seed in {1..5}  # Run experiment on 4 datasets, once per seed
                    # in total: 4 datasets × 5 seeds = 20 runs
do
    python run_models.py \
    --dataset physionet --state 'def' --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu


    python run_models.py \
    --dataset mimic --state 'def' --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu


    python run_models.py \
    --dataset activity --state 'def' --history 3000 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $seed --gpu $gpu 


    python run_models.py \
    --dataset ushcn --state 'def' --history 24 \
    --patience $patience --batch_size 192 --lr 1e-3 \
    --patch_size 2 --stride 2 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $seed --gpu $gpu
done


#| Flag           | Meaning                                                                |
#| -------------- | ---------------------------------------------------------------------- |
#| `--dataset`    | Which dataset to use (`physionet`, `mimic`, `activity`, `ushcn`)       |
#| `--state`      | Usually controls model variant — here `'def'` probably means "default" |
#| `--history`    | How much time/data to use as input history                             |
#| `--patience`   | Early stopping patience                                                |
#| `--batch_size` | Training batch size                                                    |
#| `--lr`         | Learning rate                                                          |
#| `--patch_size` | Length of patches for temporal splitting                               |
#| `--stride`     | Step size between patches                                              |
#| `--nhead`      | Number of attention heads                                              |
#| `--tf_layer`   | Number of transformer layers                                           |
#| `--nlayer`     | Likely graph neural network (GNN) layers                               |
#| `--te_dim`     | Temporal encoding dimension                                            |
#| `--node_dim`   | Node feature dimension (in GNN)                                        |
#| `--hid_dim`    | Hidden dimension for model layers                                      |
#| `--outlayer`   | Output layer type (`Linear`, `MLP`, etc.)                              |
#| `--seed`       | Random seed for reproducibility                                        |
#| `--gpu`        | GPU ID to run the model on                                             |
