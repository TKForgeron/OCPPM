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

# Print system info
utilities.torch_utils.print_system_info()

# Setup
otc_adams_config = {
    "model_output_path": "models/OTC/efg/adams",
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
}

# %%
# Get data and dataloaders
ds_train, ds_val, ds_test = data_utils.load_datasets(
    dataset_class=EFG_SG,
    storage_path=otc_adams_config["STORAGE_PATH"],
    split_feature_storage_file=otc_adams_config["SPLIT_FEATURE_STORAGE_FILE"],
    target_label=otc_adams_config["TARGET_LABEL"],
    graph_level_target=otc_adams_config["graph_level_prediction"],
    features_dtype=otc_adams_config["features_dtype"],
    target_dtype=otc_adams_config["target_dtype"],
    subgraph_size=otc_adams_config["SUBGRAPH_SIZE"],
    train=True,
    val=True,
    test=True,
    skip_cache=otc_adams_config["skip_cache"],
)
train_loader, val_loader, test_loader = data_utils.prepare_dataloaders(
    batch_size=otc_adams_config["BATCH_SIZE"],
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    num_workers=3,
    seed_worker=functools.partial(
        utilities.torch_utils.seed_worker, state=otc_adams_config["RANDOM_SEED"]
    ),
    generator=torch.Generator().manual_seed(otc_adams_config["RANDOM_SEED"]),
)


# %% [markdown]
# ### Final hyperparameter tuning

# %%
experiment_utils.run_efg_experiment_configuration(
    model_class=AdamsGCN,
    lr=0.01,
    hidden_dim=24,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    efg_config=otc_adams_config,
)
