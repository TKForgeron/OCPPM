# %%
# DEPENDENCIES
# Python native
import functools

# EFG data handling
import ocpa.algo.predictive_monitoring.factory as feature_factory

# PyTorch
import torch
import torch.optim as O

import utilities.torch_utils

# Custom imports
from experiments.efg import EFG

# from experiments.efg_sg import EFG_SG
from models.definitions.geometric_models import HigherOrderGNN
from utilities import data_utils, experiment_utils

# Print system info
utilities.torch_utils.print_system_info()

# Setup
cs_efg_config = {
    "model_output_path": "models/CS/efg",
    "STORAGE_PATH": "data/CS/feature_encodings/EFG/efg",
    "SPLIT_FEATURE_STORAGE_FILE": "CS_split_[C2_P2_P3_O3_eas].fs",
    "TARGET_LABEL": (feature_factory.EVENT_REMAINING_TIME, ()),
    "regression_task": True,
    "graph_level_prediction": False,
    "features_dtype": torch.float32,
    "target_dtype": torch.float32,
    "SUBGRAPH_SIZE": 4,
    "BATCH_SIZE": 16,
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
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "verbose": True,
    "track_time": True,
    "skip_cache": False,
    "squeeze": True,
}

# ADAPTATIONS
cs_efg_config["skip_cache"] = True

# %%
# DATA PREPARATION
ds_train, ds_val, ds_test = data_utils.load_datasets(
    dataset_class=EFG,
    storage_path=cs_efg_config["STORAGE_PATH"],
    split_feature_storage_file=cs_efg_config["SPLIT_FEATURE_STORAGE_FILE"],
    target_label=cs_efg_config["TARGET_LABEL"],
    graph_level_target=cs_efg_config["graph_level_prediction"],
    features_dtype=cs_efg_config["features_dtype"],
    target_dtype=cs_efg_config["target_dtype"],
    subgraph_size=cs_efg_config["SUBGRAPH_SIZE"],
    train=True,
    val=True,
    test=True,
    skip_cache=cs_efg_config["skip_cache"],
)
train_loader, val_loader, test_loader = data_utils.prepare_dataloaders(
    batch_size=cs_efg_config["BATCH_SIZE"],
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    num_workers=3,
    seed_worker=functools.partial(
        utilities.torch_utils.seed_worker, state=cs_efg_config["RANDOM_SEED"]
    ),
    generator=torch.Generator().manual_seed(cs_efg_config["RANDOM_SEED"]),
)

# %% [markdown]
# ### Final hyperparameter tuning
print()
print("Running hyperparameter tuning process for EFG on Financial Institution OCEL")

cs_efg_config["verbose"] = False
cs_efg_config["model_output_path"] = "models/CS/efg/exp_v3/no_subgraph_sampling"
# CHECK BATCH SIZE

lr_range = [0.01, 0.001]
hidden_dim_range = [8, 16, 24, 32, 48, 64, 128, 256]
for lr in lr_range:
    for hidden_dim in hidden_dim_range:
        experiment_utils.run_efg_experiment_configuration(
            model_class=HigherOrderGNN,
            lr=lr,
            hidden_dim=hidden_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            efg_config=cs_efg_config,
        )
