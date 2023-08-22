# %%
# DEPENDENCIES
# Python native
import functools
import json
import pickle
import random
from copy import copy
from datetime import datetime
from pprint import pprint
from statistics import median as median
from sys import platform
from typing import Any, Callable, Union

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
# Custom imports
# from loan_application_experiment.feature_encodings.efg.efg import EFG
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import utilities.torch_utils
from experiments.hoeg import HOEG
# from importing_ocel import build_feature_storage, load_ocel, pickle_feature_storage
from models.definitions.geometric_models import (GraphModel,
                                                 HeteroHigherOrderGNN)
from utilities import (evaluation_utils, hetero_data_utils,
                       hetero_evaluation_utils, hetero_experiment_utils,
                       hetero_training_utils)

# Print system info
utilities.torch_utils.print_system_info()
utilities.torch_utils.print_torch_info()

# INITIAL CONFIGURATION
otc_hoeg_config = {
    "model_output_path": "models/OTC/hoeg",
    "STORAGE_PATH": "data/OTC/feature_encodings/HOEG/hoeg",
    "SPLIT_FEATURE_STORAGE_FILE": "OTC_split_[C2_P2_P3_O3_eas].fs",
    "OBJECTS_DATA_DICT": "otc_ofg+oi_graph+item_node_map+order_node_map+packages_node_map.pkl",
    "events_target_label": (feature_factory.EVENT_REMAINING_TIME, ()),
    "objects_target_label": "@@object_lifecycle_duration",
    "regression_task": True,
    "target_node_type": "event",
    "object_types": ["item", "order", "package"],
    "meta_data": (
        ["event", "item", "order", "package"],
        [
            ("event", "follows", "event"),
            ("event", "interacts", "order"),
            ("event", "interacts", "item"),
            ("event", "interacts", "package"),
        ],
    ),
    "BATCH_SIZE": 16,
    "RANDOM_SEED": 42,
    "EPOCHS": 32,
    "early_stopping": 4,
    "hidden_dim": 256,
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
    "skip_cache": False,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "squeeze": True,
    "track_time": True,
}

# CONFIGURATION ADAPTATIONS may be set here
# otc_hoeg_config['skip_cache'] = True

# %%
# DATA PREPARATION
transformations = [
    hetero_data_utils.ToUndirected(
        exclude_edge_types=[("event", "follows", "event")]
    ),  # Convert heterogeneous graphs to undirected graphs, but exclude event-event relations
    # T.ToUndirected(),  # Convert the graphs to undirected graphs
    T.AddSelfLoops(),  # Add self-loops to the graphs
    T.NormalizeFeatures(),  # Normalize node features of the graphs
]
# Get data and dataloaders
ds_train, ds_val, ds_test = hetero_data_utils.load_hetero_datasets(
    storage_path=otc_hoeg_config["STORAGE_PATH"],
    split_feature_storage_file=otc_hoeg_config["SPLIT_FEATURE_STORAGE_FILE"],
    objects_data_file=otc_hoeg_config["OBJECTS_DATA_DICT"],
    event_node_label_key=otc_hoeg_config["events_target_label"],
    object_nodes_label_key=otc_hoeg_config["objects_target_label"],
    edge_types=otc_hoeg_config["meta_data"][1],
    object_node_types=otc_hoeg_config["object_types"],
    graph_level_target=False,
    transform=T.Compose(transformations),
    train=True,
    val=True,
    test=True,
    skip_cache=otc_hoeg_config["skip_cache"],
)

# %%
# Update meta data (it has changed after applying `transformations`)
otc_hoeg_config["meta_data"] = ds_val[0].metadata()
# print_hetero_dataset_summaries(ds_train, ds_val, ds_test)
(
    train_loader,
    val_loader,
    test_loader,
) = hetero_data_utils.hetero_dataloaders_from_datasets(
    batch_size=otc_hoeg_config["BATCH_SIZE"],
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    num_workers=3,
    seed_worker=functools.partial(
        utilities.torch_utils.seed_worker, state=otc_hoeg_config["RANDOM_SEED"]
    ),
    generator=torch.Generator().manual_seed(otc_hoeg_config["RANDOM_SEED"]),
)

# %%
otc_hoeg_config["verbose"] = False
otc_hoeg_config["squeeze"] = True
otc_hoeg_config["model_output_path"] = "models/OTC/hoeg/exp_v1"

lr_range = [0.01, 0.001]
hidden_dim_range = [8, 16, 24, 32, 48, 64, 128, 256]
for lr in lr_range:
    for hidden_dim in hidden_dim_range:
        hetero_experiment_utils.run_hoeg_experiment_configuration(
            HeteroHigherOrderGNN,
            lr=lr,
            hidden_dim=hidden_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            hoeg_config=otc_hoeg_config,
        )
