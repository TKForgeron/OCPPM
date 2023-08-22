# %%
import os

go_up_n_directories = lambda path, n: os.path.abspath(
    os.path.join(*([os.path.dirname(path)] + [".."] * n))
)
os.chdir(go_up_n_directories(os.getcwd(), 3))  # run once (otherwise restart kernel)
print(os.getcwd())

# %%
# DEPENDENCIES
# Python native
import functools
import json
import os
from datetime import datetime
from statistics import median as median

# Data handling
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
from torch_geometric.data import HeteroData

import utilities.torch_utils
from experiments.hoeg import HOEG
# Custom imports
from models.definitions.geometric_models import (GraphModel,
                                                 HeteroHigherOrderGNN)
from utilities import (evaluation_utils, hetero_data_utils,
                       hetero_evaluation_utils, hetero_experiment_utils,
                       hetero_training_utils)

# Print system info
utilities.torch_utils.print_system_info()
utilities.torch_utils.print_torch_info()

# INITIAL CONFIGURATION
bpi17_hoeg_config = {
    "model_output_path": "models/BPI17/hoeg",
    "STORAGE_PATH": "data/BPI17/feature_encodings/HOEG/hoeg",
    "SPLIT_FEATURE_STORAGE_FILE": "BPI_split_[C2_P2_P3_P5_O3_Action_EventOrigin_OrgResource].fs",
    "OBJECTS_DATA_DICT": "bpi17_ofg+oi_graph+app_node_map+off_node_map.pkl",
    "events_target_label": (feature_factory.EVENT_REMAINING_TIME, ()),
    "objects_target_label": "@@object_lifecycle_duration",
    "target_node_type": "event",
    "object_types": ["application", "offer"],
    "meta_data": (
        ["event", "application", "offer"],
        [
            ("event", "follows", "event"),
            ("event", "interacts", "application"),
            ("event", "interacts", "offer"),
        ],
    ),
    "BATCH_SIZE": 16,
    "RANDOM_SEED": 42,
    "EPOCHS": 32,
    "early_stopping": 8,
    "hidden_dim": 32,
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
    "track_time": True,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "squeeze": True,
}

# CONFIGURATION ADAPTATIONS may be set here
# bpi17_hoeg_config["EPOCHS"] = 1
# bpi17_hoeg_config["early_stopping"] = 4
# bpi17_hoeg_config["skip_cache"] = True


# %%
# DATA PREPARATION
transformations = [
    hetero_data_utils.ToUndirected(
        exclude_edge_types=[("event", "follows", "event")]
    ),  # Convert heterogeneous graphs to undirected graphs, but exclude event-event relations
    # T.ToUndirected(),  # Convert the graph to an undirected graph   # this was in HOEG.py in v0.18
    T.AddSelfLoops(),  # Add self-loops to the graph                # this was in HOEG.py in v0.18
    T.NormalizeFeatures(),  # Normalize node features of the graph  # this was in HOEG.py in v0.18
]
# Get data and dataloaders
ds_train, ds_val, ds_test = hetero_data_utils.load_hetero_datasets(
    bpi17_hoeg_config["STORAGE_PATH"],
    bpi17_hoeg_config["SPLIT_FEATURE_STORAGE_FILE"],
    bpi17_hoeg_config["OBJECTS_DATA_DICT"],
    event_node_label_key=bpi17_hoeg_config["events_target_label"],
    object_nodes_label_key=bpi17_hoeg_config["objects_target_label"],
    edge_types=bpi17_hoeg_config["meta_data"][1],
    object_node_types=bpi17_hoeg_config["object_types"],
    graph_level_target=False,
    transform=T.Compose(transformations),
    train=True,
    val=True,
    test=True,
    skip_cache=bpi17_hoeg_config["skip_cache"],
)
bpi17_hoeg_config["meta_data"] = ds_val[0].metadata()
# hetero_data_utils.print_hetero_dataset_summaries(ds_train, ds_val,ds_test)
# %%
(
    train_loader,
    val_loader,
    test_loader,
) = hetero_data_utils.hetero_dataloaders_from_datasets(
    batch_size=bpi17_hoeg_config["BATCH_SIZE"],
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    num_workers=3,
    seed_worker=functools.partial(
        utilities.torch_utils.seed_worker, state=bpi17_hoeg_config["RANDOM_SEED"]
    ),
    generator=torch.Generator().manual_seed(bpi17_hoeg_config["RANDOM_SEED"]),
)


# %%
# FINAL HYPERPARAMETER TUNING
bpi17_hoeg_config["verbose"] = False
bpi17_hoeg_config["model_output_path"] = "models/BPI17/hoeg/exp_v2"

lr_range = [0.01, 0.001]
hidden_dim_range = [8, 16, 24, 32, 48, 64, 128, 256]
# lr_range = [0.001]
# hidden_dim_range = [128, 256]
for lr in lr_range:
    for hidden_dim in hidden_dim_range:
        hetero_experiment_utils.run_hoeg_experiment_configuration(
            HeteroHigherOrderGNN,
            lr=lr,
            hidden_dim=hidden_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            hoeg_config=bpi17_hoeg_config,
        )
