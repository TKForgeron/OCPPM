# %%
# DEPENDENCIES
# Python native
import functools
import json
import os

os.chdir("/home/tim/Development/OCPPM/")
import logging
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
from ocpa.algo.predictive_monitoring.obj import \
    Feature_Storage as FeatureStorage
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
import utilities.torch_utils
# Custom imports
# from loan_application_experiment.feature_encodings.efg.efg import EFG
from experiments.hoeg import HOEG
# from importing_ocel import build_feature_storage, load_ocel, pickle_feature_storage
from models.definitions.geometric_models import (GraphModel,
                                                 HeteroHigherOrderGNN)

# Print system info
utilities.torch_utils.print_system_info()
utilities.torch_utils.print_torch_info()

# INITIAL CONFIGURATION
cs_hoeg_config = {
    "model_output_path": "models/CS/hoeg",
    "STORAGE_PATH": "data/CS/feature_encodings/HOEG/hoeg",
    "SPLIT_FEATURE_STORAGE_FILE": "CS_split_[C2_P2_P3_O3_eas].fs",
    "events_target_label": (feature_factory.EVENT_REMAINING_TIME, ()),
    "objects_target_label": "@@object_lifecycle_duration",
    "OBJECTS_DATA_DICT": "cs_ofg+oi_graph+krs_node_map+krv_node_map+cv_node_map.pkl",
    "BATCH_SIZE": 16,
    "RANDOM_SEED": 42,
    "EPOCHS": 32,
    "target_node_type": "event",
    "object_types":["krs", "krv", "cv"],
    "meta_data": (
        ["event", "krs", "krv", "cv"],
        [
            ("event", "follows", "event"),
            ("event", "interacts", "krs"),
            ("event", "interacts", "krv"),
            ("event", "interacts", "cv"),
        ],
    ),
    "early_stopping": 8,
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

# CONFIGURATION ADAPTATIONS may be set here
# cs_hoeg_config["early_stopping"] = 4
cs_hoeg_config['skip_cache'] = True
cs_hoeg_config['verbose'] = False
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logging/debug.log",
)
logging.critical("-" * 32 + ' TEST CS HOEG ' + "-" * 32)

# %%
# DATA PREPARATION
transformations = [
    T.ToUndirected(),  # Convert the graph to an undirected graph
    T.AddSelfLoops(),  # Add self-loops to the graph
    T.NormalizeFeatures(),  # Normalize node features of the graph
]
# Get data and dataloaders
ds_train, ds_val, ds_test = hetero_data_utils.load_hetero_datasets(
    storage_path= cs_hoeg_config["STORAGE_PATH"],
    split_feature_storage_file= cs_hoeg_config["SPLIT_FEATURE_STORAGE_FILE"],
    objects_data_file=cs_hoeg_config["OBJECTS_DATA_DICT"],
    event_node_label_key= cs_hoeg_config["events_target_label"],
    object_nodes_label_key=cs_hoeg_config['objects_target_label'],
    edge_types=cs_hoeg_config['meta_data'][1],
    object_node_types=cs_hoeg_config['object_types'],
    graph_level_target = False,
    transform=T.Compose(transformations),
    train=True,
    val=True,
    test=True,
    verbosity=0,
    skip_cache=cs_hoeg_config["skip_cache"],
)
# update meta data
cs_hoeg_config["meta_data"] = ds_val[0].metadata()
# print_hetero_dataset_summaries(ds_train, ds_val, ds_test)
(
    train_loader,
    val_loader,
    test_loader,
) = hetero_data_utils.hetero_dataloaders_from_datasets(
    batch_size=cs_hoeg_config["BATCH_SIZE"],
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    seed_worker=functools.partial(
        utilities.torch_utils.seed_worker, state=cs_hoeg_config["RANDOM_SEED"]
    ),
    generator=torch.Generator().manual_seed(cs_hoeg_config["RANDOM_SEED"]),
)

# %%
# MODEL INITIATION
model = HeteroHigherOrderGNN(32, 1)
model = pygnn.to_hetero(model, cs_hoeg_config["meta_data"])
model.to(cs_hoeg_config["device"])

# Print summary of data and model
if cs_hoeg_config["verbose"]:
    print(model)
    with torch.no_grad():  # Initialize lazy modules, s.t. we can count its parameters.
        batch = next(iter(train_loader))
        batch.to(cs_hoeg_config["device"])
        out = model(batch.x_dict, batch.edge_index_dict)
        print(f"Number of parameters: {utilities.torch_utils.count_parameters(model)}")

# %%
# MODEL TRAINING
print("Training started, progress available in Tensorboard")
torch.cuda.empty_cache()

timestamp = datetime.now().strftime("%Y%m%d_%Hh%Mm")
model_path_base = f"{cs_hoeg_config['model_output_path']}/{str(model).split('(')[0]}_{timestamp}"

best_state_dict_path = hetero_training_utils.run_training_hetero(
    target_node_type=cs_hoeg_config["target_node_type"],
    num_epochs=cs_hoeg_config["EPOCHS"],
    model=model,
    train_loader=train_loader,
    validation_loader=val_loader,
    optimizer=O.Adam(model.parameters(), **cs_hoeg_config["optimizer_settings"]),
    loss_fn=cs_hoeg_config["loss_fn"],
    early_stopping_criterion=cs_hoeg_config["early_stopping"],
    model_path_base=model_path_base,
    device=cs_hoeg_config["device"],
    verbose=False,
)

# Write experiment settings as JSON into model path (of the model we've just trained)
with open(os.path.join(model_path_base, "experiment_settings.json"), "w") as file_path:
    json.dump(evaluation_utils.get_json_serializable_dict(cs_hoeg_config), file_path)

# %%
# MODEL EVALUATION
state_dict_path = f"{cs_hoeg_config['model_output_path']}/GraphModule_20230718_16h54m"  # 0.3902 test mae | 21k params (I DO NOT BELIEVE IT)

# Get MAE results
evaluation_dict = hetero_evaluation_utils.evaluate_best_model(
    target_node_type=cs_hoeg_config["target_node_type"],
    model_state_dict_path=best_state_dict_path,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model=model,
    metric=torch.nn.L1Loss(),
    device=cs_hoeg_config["device"],
    verbose=cs_hoeg_config["verbose"],
)

# Store model results as JSON into model path
with open(os.path.join(model_path_base, "evaluation_report.json"), "w") as file_path:
    json.dump(evaluation_utils.get_json_serializable_dict(evaluation_dict), file_path)

# Print MAE results
print(model_path_base)
print(evaluation_dict)


