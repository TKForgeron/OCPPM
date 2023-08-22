# %%
# DEPENDENCIES
import functools
import json
import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as O
import torch.utils.tensorboard
import torch_geometric.loader as L
import torch_geometric.nn as pygnn
import torch_geometric.transforms as T
from torch_geometric.data import Data, HeteroData

import utilities.evaluation_utils as evaluation_utils
import utilities.hetero_data_utils as hetero_data_utils
import utilities.hetero_evaluation_utils as hetero_evaluation_utils
import utilities.hetero_training_utils as hetero_training_utils
import utilities.torch_utils as torch_utils

# Custom imports
from models.definitions.geometric_models import GraphModel, HeteroHigherOrderGNN

# Print system info
torch_utils.print_system_info()
torch_utils.print_torch_info()

# INITIAL CONFIGURATION
bpi17_ofg_config = {
    "ofg_file": "data/BPI17/feature_encodings/OFG/ofg/raw/BPI17_OFG.pkl",
    "model_output_path": "models/BPI17/ofg",
    "BATCH_SIZE": 128,
    "RANDOM_SEED": 42,
    "EPOCHS": 30,
    "target_node_type": "application",
    "meta_data": (
        ["application", "offer"],
        [
            ("application", "interacts", "application"),
            ("application", "interacts", "offer"),
            ("offer", "interacts", "offer"),
            ("offer", "rev_interacts", "application"),
        ],
    ),
    "early_stopping": 3,
    "optimizer_settings": {
        "lr": 1e-3,
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

# CONFIGURATION ADAPTATIONS may be set here
# bpi17_hoeg_config["early_stopping"] = 4

# %%
# DATA PREPARATION
# Load HeteroData object from a pickle file using the specified file path
with open(bpi17_ofg_config["ofg_file"], "rb") as fp:
    data: HeteroData = pickle.load(fp)
# Define a list of transformations to be applied in sequence
torch.manual_seed(bpi17_ofg_config["RANDOM_SEED"])
transformations = [
    T.ToUndirected(),  # Convert the graph to an undirected graph
    T.AddSelfLoops(),  # Add self-loops to the graph
    T.NormalizeFeatures(),  # Normalize node features of the graph
    T.RandomNodeSplit(
        num_val=0.8 * 0.2, num_test=0.2
    ),  # Split the graph into train, validation, and test sets based on random node assignment
]
# Apply the transformation pipeline to the data at once
data = T.Compose(transformations)(data)
bpi17_ofg_config["meta_data"] = data.metadata()
# Create hetero dataloaders for each split
(
    train_loader,
    val_loader,
    test_loader,
) = hetero_data_utils.hetero_dataloaders_from_hetero_data(
    hetero_data=data,
    batch_size=bpi17_ofg_config["BATCH_SIZE"],
    num_neighbors=[3] * 2,
    node_type=bpi17_ofg_config["target_node_type"],
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    generator=torch.Generator().manual_seed(bpi17_ofg_config["RANDOM_SEED"]),
)

# %%
# MODEL INITIATION
model = HeteroHigherOrderGNN(64, 1)
model = pygnn.to_hetero(model, bpi17_ofg_config["meta_data"])
model.double()
model.to(bpi17_ofg_config["device"])

# Print summary of data and model
if bpi17_ofg_config["verbose"]:
    # print(model)
    with torch.no_grad():  # Initialize lazy modules, s.t. we can count its parameters.
        batch = next(iter(train_loader))
        batch.to(bpi17_ofg_config["device"])
        out = model(batch.x_dict, batch.edge_index_dict)
        print(f"Number of parameters: {torch_utils.count_parameters(model)}")

# %%
# MODEL TRAINING
print("Training started, progress available in Tensorboard")
torch.cuda.empty_cache()

timestamp = datetime.now().strftime("%Y%m%d_%Hh%Mm")
model_path_base = (
    f"{bpi17_ofg_config['model_output_path']}/{str(model).split('(')[0]}_{timestamp}"
)

best_state_dict_path = hetero_training_utils.run_training_hetero(
    target_node_type=bpi17_ofg_config["target_node_type"],
    num_epochs=bpi17_ofg_config["EPOCHS"],
    model=model,
    train_loader=train_loader,
    validation_loader=val_loader,
    optimizer=O.Adam(model.parameters(), **bpi17_ofg_config["optimizer_settings"]),
    loss_fn=bpi17_ofg_config["loss_fn"],
    early_stopping_criterion=bpi17_ofg_config["early_stopping"],
    model_path_base=model_path_base,
    device=bpi17_ofg_config["device"],
    verbose=False,
)

# Write experiment settings as JSON into model path (of the model we've just trained)
with open(os.path.join(model_path_base, "experiment_settings.json"), "w") as file_path:
    json.dump(evaluation_utils.get_json_serializable_dict(bpi17_ofg_config), file_path)

# %%
state_dict_path = f"{bpi17_ofg_config['model_output_path']}/GraphModule_20230724_11h33m/state_dict_epoch1.pt"  # 0.7328 test mae | HigherOrderGNN_HOEG(32, 1) | 10k params
state_dict_path = f"{bpi17_ofg_config['model_output_path']}/GraphModule_20230724_11h39m/state_dict_epoch1.pt"  # 0.7120 test mae | HigherOrderGNN_HOEG(64, 1) | 36k params

# Get MAE results
evaluation_dict = hetero_evaluation_utils.evaluate_best_model(
    target_node_type=bpi17_ofg_config["target_node_type"],
    model_state_dict_path=best_state_dict_path,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model=model,
    metric=torch.nn.L1Loss(),
    device=bpi17_ofg_config["device"],
    verbose=bpi17_ofg_config["verbose"],
)

# Store model results as JSON into model path
with open(os.path.join(model_path_base, "evaluation_report.json"), "w") as file_path:
    json.dump(evaluation_utils.get_json_serializable_dict(evaluation_dict), file_path)

# Print MAE results
print(model_path_base)
print(evaluation_dict)
