# %%
# DEPENDENCIES
# Python native
import functools

# Data handling
import ocpa.algo.predictive_monitoring.factory as feature_factory

# PyG
import torch
import torch.optim as O

import utilities.torch_utils

# Custom imports
from experiments.efg_sg import EFG_SG
from models.definitions.geometric_models import AdamsGCN
from utilities import data_utils, experiment_utils

# Setup
cs_adams_config = {
    "model_output_path": "models/CS/baselines/adams",
    "STORAGE_PATH": "data/CS/feature_encodings/EFG/efg",
    "SPLIT_FEATURE_STORAGE_FILE": "CS_split_[C2_P2_P3_O3_eas].fs",
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
    "hidden_dim": 24,
    "optimizer": O.Adam,
    "optimizer_settings": {
        "lr": 0.01,
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
    "squeeze": True,
}

# CONFIGURATION ADAPTATIONS may be set here
# cs_adams_config['skip_cache'] = True

# Print system info
if cs_adams_config["verbose"]:
    utilities.torch_utils.print_system_info()

# %%
# Get data and dataloaders
ds_train, ds_val, ds_test = data_utils.load_datasets(
    dataset_class=EFG_SG,
    storage_path=cs_adams_config["STORAGE_PATH"],
    split_feature_storage_file=cs_adams_config["SPLIT_FEATURE_STORAGE_FILE"],
    target_label=cs_adams_config["TARGET_LABEL"],
    graph_level_target=cs_adams_config["graph_level_prediction"],
    features_dtype=cs_adams_config["features_dtype"],
    target_dtype=cs_adams_config["target_dtype"],
    subgraph_size=cs_adams_config["SUBGRAPH_SIZE"],
    train=True,
    val=True,
    test=True,
    skip_cache=cs_adams_config["skip_cache"],
)
train_loader, val_loader, test_loader = data_utils.prepare_dataloaders(
    batch_size=cs_adams_config["BATCH_SIZE"],
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    num_workers=3,
    seed_worker=functools.partial(
        utilities.torch_utils.seed_worker, state=cs_adams_config["RANDOM_SEED"]
    ),
    generator=torch.Generator().manual_seed(cs_adams_config["RANDOM_SEED"]),
)

print()
print(
    "Running Adams et al. (2022) replication for baseline experiment on Financial Institution OCEL"
)

cs_adams_config["verbose"] = False

experiment_utils.run_efg_experiment_configuration(
    model_class=AdamsGCN,
    lr=cs_adams_config["optimizer_settings"]["lr"],
    hidden_dim=cs_adams_config["hidden_dim"],
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    efg_config=cs_adams_config,
)
