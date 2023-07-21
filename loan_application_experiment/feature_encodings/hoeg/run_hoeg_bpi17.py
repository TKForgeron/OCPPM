# %%
# DEPENDENCIES
# Python native
# os.chdir("/home/tim/Development/OCPPM/")
import functools
import json
import os
import pickle
import random
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

import utilities.evaluation_utils as evaluation_utils
import utilities.hetero_data_utils as hetero_data_utils
import utilities.hetero_evaluation_utils as hetero_evaluation_utils
import utilities.hetero_training_utils as hetero_training_utils
import utilities.torch_utils as torch_utils

# Custom imports
# from loan_application_experiment.feature_encodings.efg.efg import EFG
from loan_application_experiment.feature_encodings.hoeg.hoeg import HOEG

# from importing_ocel import build_feature_storage, load_ocel, pickle_feature_storage
from loan_application_experiment.models.geometric_models import (
    AGNN_EFG,
    AdamsGCN,
    GraphModel,
    HigherOrderGNN_EFG,
)

# Print system info
torch_utils.print_system_info()
torch_utils.print_torch_info()

# INITIAL CONFIGURATION
bpi17_hoeg_config = {
    "STORAGE_PATH": "data/BPI17/feature_encodings/HOEG/hoeg",
    "SPLIT_FEATURE_STORAGE_FILE": "BPI_split_[C2_P2_P3_P5_O3_Action_EventOrigin_OrgResource].fs",
    "TARGET_LABEL": (feature_factory.EVENT_REMAINING_TIME, ()),
    "OBJECTS_DATA_DICT": "bpi17_ofg+oi_graph+app_node_map+off_node_map.pkl",
    "BATCH_SIZE": 64,
    "RANDOM_SEED": 42,
    "EPOCHS": 30,
    "meta_data": (
        ["event", "application", "offer"],
        [
            ("event", "follows", "event"),
            ("event", "interacts", "application"),
            ("event", "interacts", "offer"),
            ("application", "interacts", "application"),
            ("application", "rev_interacts", "event"),
            ("offer", "rev_interacts", "event"),
        ],
    ),
    "early_stopping": 5,
    "optimizer_settings": {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": False,
    },
    "loss_fn": torch.nn.L1Loss(),
    "verbose": True,
    "skip_cache": False,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# CONFIGURATION ADAPTATIONS
bpi17_hoeg_config["BATCH_SIZE"] = 16
bpi17_hoeg_config["EPOCHS"] = 32
bpi17_hoeg_config["early_stopping"] = 8
bpi17_hoeg_config["optimizer_settings"] = {
    "lr": 5e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0,
    "amsgrad": False,
}
# bpi17_hoeg_config["skip_cache"] = True
# bpi17_hoeg_config["device"] = torch.device("cpu")


# %%
# DATA PREPARATION
# Get data and dataloaders
ds_train, ds_val, ds_test = hetero_data_utils.load_hetero_datasets(
    bpi17_hoeg_config["STORAGE_PATH"],
    bpi17_hoeg_config["SPLIT_FEATURE_STORAGE_FILE"],
    bpi17_hoeg_config["OBJECTS_DATA_DICT"],
    bpi17_hoeg_config["TARGET_LABEL"],
    transform=T.ToUndirected(),
    train=True,
    val=True,
    test=True,
    skip_cache=bpi17_hoeg_config["skip_cache"],
)
# print_hetero_dataset_summaries(ds_train, ds_val,ds_test)
train_loader, val_loader, test_loader = hetero_data_utils.prepare_hetero_dataloaders(
    batch_size=bpi17_hoeg_config["BATCH_SIZE"],
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    seed_worker=functools.partial(
        torch_utils.seed_worker, state=bpi17_hoeg_config["RANDOM_SEED"]
    ),
    generator=torch.Generator().manual_seed(bpi17_hoeg_config["RANDOM_SEED"]),
)


# %%
# MODEL DEFINITION
class HeteroGraphConvNet(GraphModel):
    """Implementation of a Attentional Graph Neural Network for EFG"""

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = pygnn.GraphConv(-1, hidden_channels)
        self.conv2 = pygnn.GraphConv(-1, hidden_channels)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        # self.pool1 = pygnn.global_mean_pool
        self.lin_out = pygnn.Linear(-1, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = self.conv2(x, edge_index)
        x = self.act2(x)
        # x = self.pool1(x, batch)
        x = self.lin_out(x)
        return x


model = HeteroGraphConvNet(32, 1)
model = pygnn.to_hetero(model, bpi17_hoeg_config["meta_data"])
model.to(bpi17_hoeg_config["device"])

# Print summary of data and model
if bpi17_hoeg_config["verbose"]:
    print(model)
    with torch.no_grad():  # Initialize lazy modules, s.t. we can count its parameters.
        batch = next(iter(train_loader))
        batch.to(bpi17_hoeg_config["device"])
        out = model(batch.x_dict, batch.edge_index_dict, batch["event"].batch)
        print(f"Number of parameters: {torch_utils.count_parameters(model)}")

# %%
# MODEL TRAINING
print("Training started, progress available in Tensorboard")
torch.cuda.empty_cache()

timestamp = datetime.now().strftime("%Y%m%d_%Hh%Mm")
model_path_base = f"models/BPI17/hoeg/{str(model).split('(')[0]}_{timestamp}"

best_state_dict_path = hetero_training_utils.run_training_hetero(
    target_node_type="event",
    num_epochs=bpi17_hoeg_config["EPOCHS"],
    model=model,
    train_loader=train_loader,
    validation_loader=val_loader,
    optimizer=O.Adam(model.parameters(), **bpi17_hoeg_config["optimizer_settings"]),
    loss_fn=bpi17_hoeg_config["loss_fn"],
    early_stopping_criterion=bpi17_hoeg_config["early_stopping"],
    model_path_base=model_path_base,
    device=bpi17_hoeg_config["device"],
    verbose=False,
)

# Write experiment settings as JSON into model path (of the model we've just trained)
with open(os.path.join(model_path_base, "experiment_settings.json"), "w") as file_path:
    json.dump(evaluation_utils.get_json_serializable_dict(bpi17_hoeg_config), file_path)

# %%
# MODEL EVALUATION
state_dict_path = "models/BPI17/hoeg/GraphModule_20230718_16h54m"  # 0.3902 test mae | 21k params (I DO NOT BELIEVE IT)
state_dict_path = (
    "models/BPI17/hoeg/GraphModule_20230718_17h02m"  # 0.4182 test mae | 21k params
)
state_dict_path = (
    "models/BPI17/hoeg/GraphModule_20230718_17h07m"  # 0.4354 test mae | 21k params
)
state_dict_path = "models/BPI17/hoeg/GraphModule_20230719_18h06m"  # 0.2251 test mae | HeteroGraphConvNet(32, 1) | 21k params // best so far! (reloading model, re-evaluating: same result)
state_dict_path = "models/BPI17/hoeg/GraphModule_20230719_18h52m"  # 0.2185 test mae | HeteroGraphConvNet(32, 1) | 21k params (exact re-run of previous model)
state_dict_path = "models/BPI17/hoeg/GraphModule_20230720_14h05m/state_dict_epoch2.pt"  # 0.2287 test mae | HeteroGraphConvNet(32, 1) | 21k params (corrected for object_lifecycle_duration in X of object nodes)
state_dict_path = "models/BPI17/hoeg/GraphModule_20230720_14h20m/state_dict_epoch3.pt"  # 0.1919 test mae | (exact re-run of previous model)
state_dict_path = "models/BPI17/hoeg/GraphModule_20230720_14h56m/state_dict_epoch0.pt"  # 0.3441 test mae | HeteroGraphConvNet(27, 1) | 21k params (corrected for object_lifecycle_duration in X of object nodes)

# Get MAE results
evaluation_dict = hetero_evaluation_utils.evaluate_best_model(
    target_node_type="event",
    model_state_dict_path=best_state_dict_path,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model=model,
    metric=torch.nn.L1Loss(),
    device=bpi17_hoeg_config["device"],
    verbose=bpi17_hoeg_config["verbose"],
)

# Store model results as JSON into model path
with open(os.path.join(model_path_base, "evaluation_report.json"), "w") as file_path:
    json.dump(evaluation_utils.get_json_serializable_dict(evaluation_dict), file_path)

# Print MAE results
print(model_path_base)
print(evaluation_dict)
