import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

import lib.utils as utils
from torch.distributions import uniform
from torch.utils.data import DataLoader
from sklearn import model_selection

# datasets
from lib.physionet import (
    PhysioNet,
    variable_time_collate_fn as PHYS_VCOLLATE,
    patch_variable_time_collate_fn as PHYS_PCOLLATE,
    get_data_min_max as PHYS_GET_MINMAX,
)
from lib.mimic import MIMIC
from lib.ushcn import (
    USHCN,
    USHCN_time_chunk,
    USHCN_variable_time_collate_fn as USHCN_VCOLLATE,
    USHCN_patch_variable_time_collate_fn as USHCN_PCOLLATE,
    USHCN_get_seq_length,
)
from lib.person_activity import (
    PersonActivity,
    Activity_time_chunk,
    #variable_time_collate_fn as ACT_VCOLLATE,
    #patch_variable_time_collate_fn as ACT_PCOLLATE,
    Activity_get_seq_length,
)

#####################################################################################################
def parse_datasets(args, patch_ts=False, length_stat=False):
    """
    Builds data loaders + metadata for the selected dataset.
    Returns a dict:
      {
        "train_dataloader", "val_dataloader", "test_dataloader",
        "n_train_batches", "n_val_batches", "n_test_batches",
        "input_dim", "data_min", "data_max", "time_max",
        (optional) "max_input_len", "max_pred_len", "median_len"
      }
    """
    device = args.device
    dataset_name = str(args.dataset).lower()

    # ------------------------------------------------------------
    # PhysioNet & MIMIC (very similar pipelines)
    # ------------------------------------------------------------
    if dataset_name in ["physionet", "mimic"]:
        # 1) load
        if dataset_name == "physionet":
            total_dataset = PhysioNet(
                "./data/physionet",
                quantization=args.quantization,
                download=True,
                n_samples=args.n,
                device=device,
            )
        else:  # mimic
            total_dataset = MIMIC(
                "../data/mimic",
                n_samples=args.n,
                device=device,
            )

        # 2) split
        seen_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            seen_data, train_size=0.75, random_state=42, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))

        # 3) basic dims
        record_id, tt, vals, mask = train_data[0]
        input_dim = vals.size(-1)

        # 4) normalization
        data_min, data_max, time_max = PHYS_GET_MINMAX(seen_data, device)

        # 5) collate
        collate_fn = PHYS_PCOLLATE if patch_ts else PHYS_VCOLLATE
        batch_size = min(min(len(seen_data), args.batch_size), args.n)

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            collate_fn=lambda batch: collate_fn(
                batch, args, device, data_type="train",
                data_min=data_min, data_max=data_max, time_max=time_max
            )
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(
                batch, args, device, data_type="val",
                data_min=data_min, data_max=data_max, time_max=time_max
            )
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(
                batch, args, device, data_type="test",
                data_min=data_min, data_max=data_max, time_max=time_max
            )
        )

        data_objects = {
            "train_dataloader": utils.inf_generator(train_loader),
            "val_dataloader": utils.inf_generator(val_loader),
            "test_dataloader": utils.inf_generator(test_loader),
            "n_train_batches": len(train_loader),
            "n_val_batches": len(val_loader),
            "n_test_batches": len(test_loader),
            "input_dim": input_dim,
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
        }

        if length_stat:
            # for physionet/mimic we can reuse an existing helper if you have it;
            # otherwise skip or implement similarly to USHCN/Activity
            pass

        return data_objects

    # ------------------------------------------------------------
    # USHCN
    # ------------------------------------------------------------
    if dataset_name == "ushcn":
        # time in "months" (code expects this scale)
        args.n_months = 48
        args.pred_window = 1  # predict one month ahead

        # 1) load
        total_dataset = USHCN("/home/solgi/srptpatch/data/ushcn", n_samples=args.n, device=device)

        # 2) split
        seen_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            seen_data, train_size=0.75, random_state=42, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))

        # 3) dims
        record_id, tt, vals, mask = train_data[0]
        input_dim = vals.size(-1)

        # 4) normalization (min/max over seen_data)
        # (USHCN returns (n_dim,) tensors + scalar time_max as in physionet style)
        data_min, data_max, time_max = PHYS_GET_MINMAX(seen_data, device)

        # 5) chunk into rolling windows (history+pred_window)
        train_data = USHCN_time_chunk(train_data, args, device)
        val_data   = USHCN_time_chunk(val_data,   args, device)
        test_data  = USHCN_time_chunk(test_data,  args, device)
        print("Dataset n_samples after time split:",
              len(train_data)+len(val_data)+len(test_data),
              len(train_data), len(val_data), len(test_data))

        # 6) collate
        collate_fn = USHCN_PCOLLATE if patch_ts else USHCN_VCOLLATE
        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max)
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max)
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, args, device, time_max=time_max)
        )

        data_objects = {
            "train_dataloader": utils.inf_generator(train_loader),
            "val_dataloader": utils.inf_generator(val_loader),
            "test_dataloader": utils.inf_generator(test_loader),
            "n_train_batches": len(train_loader),
            "n_val_batches": len(val_loader),
            "n_test_batches": len(test_loader),
            "input_dim": input_dim,
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
        }

        if length_stat:
            max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data+val_data+test_data)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"]  = max_pred_len.item()
            data_objects["median_len"]    = median_len.item()
            print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

        return data_objects

    # ------------------------------------------------------------
    # Person Activity
    # ------------------------------------------------------------
    if dataset_name == "activity":
        # milliseconds
        args.pred_window = 1000  # predict future 1000 ms

        # 1) load
        total_dataset = PersonActivity(
            "/home/solgi/srptpatch/data/PersonActivity",
            n_samples=args.n,
            download=False,
            device=device,
        )

        # 2) split
        seen_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True
        )
        train_data, val_data = model_selection.train_test_split(
            seen_data, train_size=0.75, random_state=42, shuffle=False
        )
        print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))

        # 3) dims
        record_id, tt, vals, mask = train_data[0]
        input_dim = vals.size(-1)

        # 4) normalization (reuse phys helper; returns (data_min, data_max, _))
        data_min, data_max, _ = PHYS_GET_MINMAX(seen_data, device)
        time_max = torch.tensor(args.history + args.pred_window)
        print('manual set time_max:', time_max)

        # 5) chunk
        train_data = Activity_time_chunk(train_data, args, device)
        val_data   = Activity_time_chunk(val_data,   args, device)
        test_data  = Activity_time_chunk(test_data,  args, device)
        print("Dataset n_samples after time split:",
              len(train_data)+len(val_data)+len(test_data),
              len(train_data), len(val_data), len(test_data))

        # 6) collate
        #collate_fn = ACT_PCOLLATE if patch_ts else ACT_VCOLLATE
        collate_fn = PHYS_PCOLLATE if patch_ts else PHYS_VCOLLATE

        batch_size = args.batch_size

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            collate_fn=lambda batch: collate_fn(
                batch, args, device, data_type="train",
                data_min=data_min, data_max=data_max, time_max=time_max
            )
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(
                batch, args, device, data_type="val",
                data_min=data_min, data_max=data_max, time_max=time_max
            )
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(
                batch, args, device, data_type="test",
                data_min=data_min, data_max=data_max, time_max=time_max
            )
        )

        data_objects = {
            "train_dataloader": utils.inf_generator(train_loader),
            "val_dataloader": utils.inf_generator(val_loader),
            "test_dataloader": utils.inf_generator(test_loader),
            "n_train_batches": len(train_loader),
            "n_val_batches": len(val_loader),
            "n_test_batches": len(test_loader),
            "input_dim": input_dim,
            "data_min": data_min,
            "data_max": data_max,
            "time_max": time_max,
        }

        if length_stat:
            max_input_len, max_pred_len, median_len = Activity_get_seq_length(args, train_data+val_data+test_data)
            data_objects["max_input_len"] = max_input_len.item()
            data_objects["max_pred_len"]  = max_pred_len.item()
            data_objects["median_len"]    = median_len.item()
            print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

        return data_objects

    # ------------------------------------------------------------
    # Unknown dataset name
    # ------------------------------------------------------------
    print(f"[parse_datasets] Unknown dataset: {args.dataset}")
    return None
