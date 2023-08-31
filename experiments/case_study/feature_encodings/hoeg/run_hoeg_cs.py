# %%
# DEPENDENCIES
# Python native
import functools

# Data handling
import ocpa.algo.predictive_monitoring.factory as feature_factory

# PyG
import torch
import torch.optim as O
import torch_geometric.transforms as T

import utilities.torch_utils
from models.definitions.geometric_models import HigherOrderGNN
from utilities import hetero_data_utils, hetero_experiment_utils

# Print system info
utilities.torch_utils.print_system_info()
utilities.torch_utils.print_torch_info()

# INITIAL CONFIGURATION
cs_hoeg_config = {
    "model_output_path": "models/CS/hoeg",
    "STORAGE_PATH": "data/CS/feature_encodings/HOEG/hoeg",
    "SPLIT_FEATURE_STORAGE_FILE": "CS_split_[C2_P2_P3_O3_eas].fs",
    "OBJECTS_DATA_DICT": "cs_ofg+oi_graph+krs_node_map+krv_node_map+cv_node_map.pkl",
    "events_target_label": (feature_factory.EVENT_REMAINING_TIME, ()),
    "objects_target_label": "@@object_lifecycle_duration",
    "graph_level_target": False,
    "regression_task": True,
    "target_node_type": "event",
    "object_types": ["krs", "krv", "cv"],
    "meta_data": (
        ["event", "krs", "krv", "cv"],
        [
            ("event", "follows", "event"),
            ("krs", "interacts", "event"),
            ("krv", "interacts", "event"),
            ("cv", "interacts", "event"),
        ],
    ),
    "BATCH_SIZE": 16,
    "RANDOM_SEED": 42,
    "EPOCHS": 30,
    "early_stopping": 4,
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
# cs_hoeg_config['skip_cache'] = True

# %%
# DATA PREPARATION
transformations = [
    hetero_data_utils.AddObjectSelfLoops(),  # Prepares object-object relations, which are filled when `T.AddSelfLoops()` is executed
    T.AddSelfLoops(),  # Add self-loops to the graph
    T.NormalizeFeatures(),  # Normalize node features of the graph
]
# Get data and dataloaders
ds_train, ds_val, ds_test = hetero_data_utils.load_hetero_datasets(
    storage_path=cs_hoeg_config["STORAGE_PATH"],
    split_feature_storage_file=cs_hoeg_config["SPLIT_FEATURE_STORAGE_FILE"],
    objects_data_file=cs_hoeg_config["OBJECTS_DATA_DICT"],
    event_node_label_key=cs_hoeg_config["events_target_label"],
    object_nodes_label_key=cs_hoeg_config["objects_target_label"],
    edge_types=cs_hoeg_config["meta_data"][1],
    object_node_types=cs_hoeg_config["object_types"],
    graph_level_target=cs_hoeg_config["graph_level_target"],
    transform=T.Compose(transformations),
    train=True,
    val=True,
    test=True,
    skip_cache=cs_hoeg_config["skip_cache"],
    debug=True,
)
for data in ds_val:
    data: hetero_data_utils.HeteroData
    if data.metadata() != cs_hoeg_config["meta_data"]:
        cs_hoeg_config["meta_data"] = data.metadata()
        break
(
    train_loader,
    val_loader,
    test_loader,
) = hetero_data_utils.hetero_dataloaders_from_datasets(
    batch_size=cs_hoeg_config["BATCH_SIZE"],
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    num_workers=3,
    seed_worker=functools.partial(
        utilities.torch_utils.seed_worker, state=cs_hoeg_config["RANDOM_SEED"]
    ),
    generator=torch.Generator().manual_seed(cs_hoeg_config["RANDOM_SEED"]),
)

# %%
print()
print("Running hyperparameter tuning process for HOEG on Financial Institution OCEL")

cs_hoeg_config["verbose"] = False
cs_hoeg_config["model_output_path"] = "models/CS/hoeg/exp_v3"
# cs_hoeg_config["device"] = torch.device("cpu")

lr_range = [0.01, 0.001]
hidden_dim_range = [8, 16, 24, 32, 48, 64, 128, 256]
for lr in lr_range:
    for hidden_dim in hidden_dim_range:
        hetero_experiment_utils.run_hoeg_experiment_configuration(
            HigherOrderGNN,
            lr=lr,
            hidden_dim=hidden_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            hoeg_config=cs_hoeg_config,
        )
