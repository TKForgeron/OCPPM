# # %%
# import os

# go_up_n_directories = lambda path, n: os.path.abspath(os.path.join(*([os.path.dirname(path)] + [".."] * n)))
# os.chdir(go_up_n_directories(os.getcwd(), 3)) # run once (otherwise restart kernel)
# os.getcwd()

import functools
import json

# %%
# DEPENDENCIES
# Python native
import os
import pickle
import pprint
import random
import time
from copy import copy
from datetime import datetime
from statistics import median as median
from sys import platform
from typing import Any, Callable

# Data handling
import numpy as np
import ocpa.algo.predictive_monitoring.factory as feature_factory

# PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O

# PyTorch TensorBoard support
import torch.utils.tensorboard
import torch_geometric.nn as pygnn
import torch_geometric.transforms as T

# Object centric process mining
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage

# # Simple machine learning models, procedure tools, and evaluation metrics
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Custom imports
from experiments.efg import EFG
from experiments.efg_sg import EFG_SG

# from importing_ocel import build_feature_storage, load_ocel, pickle_feature_storage
from models.definitions.geometric_models import (
    AGNN_EFG,
    AdamsGCN,
    GraphModel,
    HigherOrderGNN_EFG,
    SimpleGNN_EFG,
)
from utilities import (
    data_utils,
    evaluation_utils,
    experiment_utils,
    torch_utils,
    training_utils,
)

# Print system info
torch_utils.print_system_info()

# Setup
otc_efg_config = {
    "model_output_path": "models/OTC/efg",
    "STORAGE_PATH": "data/OTC/feature_encodings/EFG/efg",
    "SPLIT_FEATURE_STORAGE_FILE": "OTC_split_[C2_P2_P3_O3_eas].fs",
    "TARGET_LABEL": (feature_factory.EVENT_REMAINING_TIME, ()),
    "regression_task": True,
    "graph_level_prediction": True,
    "features_dtype": torch.float32,
    "target_dtype": torch.float32,
    "SUBGRAPH_SIZE": 4,
    "BATCH_SIZE": 64,
    "RANDOM_SEED": 42,
    "EPOCHS": 30,
    "early_stopping": 4,
    "hidden_dim": 16,
    "optimizer": O.Adam,
    "optimizer_settings": {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": False,
    },
    "loss_fn": torch.nn.L1Loss(),
    "verbose": True,
    "track_time": True,
    "skip_cache": False,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

# ADAPTATIONS
# otc_efg_config['skip_cache']=True

# %%
# Get data and dataloaders
ds_train, ds_val, ds_test = data_utils.load_datasets(
    dataset_class=EFG_SG,
    storage_path=otc_efg_config["STORAGE_PATH"],
    split_feature_storage_file=otc_efg_config["SPLIT_FEATURE_STORAGE_FILE"],
    target_label=otc_efg_config["TARGET_LABEL"],
    graph_level_target=otc_efg_config["graph_level_prediction"],
    features_dtype=otc_efg_config["features_dtype"],
    target_dtype=otc_efg_config["target_dtype"],
    subgraph_size=otc_efg_config["SUBGRAPH_SIZE"],
    train=True,
    val=True,
    test=True,
    skip_cache=otc_efg_config["skip_cache"],
)
train_loader, val_loader, test_loader = data_utils.prepare_dataloaders(
    batch_size=otc_efg_config["BATCH_SIZE"],
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    num_workers=3,
    seed_worker=functools.partial(
        torch_utils.seed_worker, state=otc_efg_config["RANDOM_SEED"]
    ),
    generator=torch.Generator().manual_seed(otc_efg_config["RANDOM_SEED"]),
)

# %% [markdown]
# ### Final hyperparameter tuning


# %%
lr_range = [0.01, 0.001]
hidden_dim_range = [8, 16, 24, 32, 48, 64, 128, 256]
for lr in lr_range:
    for hidden_dim in hidden_dim_range:
        experiment_utils.run_efg_experiment_configuration(
            model_class=HigherOrderGNN_EFG,
            lr=lr,
            hidden_dim=hidden_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            otc_efg_config
        )
