import random

import numpy as np
import ocpa.algo.predictive_monitoring.factory as feature_factory
import torch

bpi17_config = {
    "STORAGE_PATH": "data/BPI17/ocpa-processed",
    "SPLIT_FEATURE_STORAGE_FILE": "BPI2017-feature_storage-split-[C1-3,C5,P1-6,O2,O3,O5].fs",
    "RAW_FEATURE_STORAGE_FILE": "BPI17-feature_storage-[C2,D1,P2,P3,O3].fs",
    "TARGET_LABEL": (feature_factory.EVENT_REMAINING_TIME, ()),
    "SUBGRAPH_SIZE": 12,
    "EPOCHS": 30,
    "BATCH_SIZE": 512,
    "RANDOM_SEED": 42,
    "ORIGINAL_LOG_FILE": "data/adams/example_logs/mdl/BPI2017-Final.csv",
    "FEATURE_SET": {
        "C2": None,
        "D1": ("event_RequestedAmount", max),
        "P2": (feature_factory.EVENT_ELAPSED_TIME, ()),
        "P3": (feature_factory.EVENT_REMAINING_TIME, ()),
        "O3": (feature_factory.EVENT_PREVIOUS_TYPE_COUNT, ("offer",)),
    },
}
# Initializing random seeds for maximizing reproducibility
torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)
np.random.seed(42)
random.seed(42)
# torch.use_deterministic_algorithms(True) # incompatible with GCN


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 42
    # worker_seed = RANDOM_SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)
