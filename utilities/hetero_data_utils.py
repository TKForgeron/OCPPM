# Python native
import os

os.chdir("/home/tim/Development/OCPPM/")

from typing import Any, Callable

# PyG
import torch

# PyTorch TensorBoard support
import torch.utils.tensorboard
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader
from tqdm import tqdm

# Custom imports
# from loan_application_experiment.feature_encodings.efg.efg import EFG
from loan_application_experiment.feature_encodings.hoeg.hoeg import HOEG

# from importing_ocel import build_feature_storage, load_ocel, pickle_feature_storage
from loan_application_experiment.models.geometric_models import GraphModel


def load_hetero_datasets(
    storage_path: str,
    split_feature_storage_file: str,
    objects_data_file: str,
    target_label: tuple[str, tuple],
    transform=None,
    train: bool = True,
    val: bool = True,
    test: bool = True,
    skip_cache: bool = False,
) -> list[HOEG]:
    datasets = []
    if train:
        ds_train = HOEG(
            train=True,
            root=storage_path,
            events_filename=split_feature_storage_file,
            objects_filename=objects_data_file,
            label_key=target_label,
            transform=transform,
            verbosity=51,
            skip_cache=skip_cache,
        )
        datasets.append(ds_train)
    if val:
        ds_val = HOEG(
            validation=True,
            root=storage_path,
            events_filename=split_feature_storage_file,
            objects_filename=objects_data_file,
            label_key=target_label,
            transform=transform,
            verbosity=51,
            skip_cache=skip_cache,
        )
        datasets.append(ds_val)
    if test:
        ds_test = HOEG(
            test=True,
            root=storage_path,
            events_filename=split_feature_storage_file,
            objects_filename=objects_data_file,
            label_key=target_label,
            transform=transform,
            verbosity=51,
            skip_cache=skip_cache,
        )
        datasets.append(ds_test)
    return datasets


def print_hetero_dataset_summaries(
    ds_train: HOEG,
    ds_val: HOEG,
    ds_test: HOEG,
) -> None:
    print("Train set")
    print(ds_train.get_summary(), "\n")
    print("Validation set")
    print(ds_val.get_summary(), "\n")
    print("Test set")
    print(ds_test.get_summary(), "\n")


def hetero_dataloaders_from_datasets(
    batch_size: int,
    ds_train: HOEG = None,
    ds_val: HOEG = None,
    ds_test: HOEG = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    num_workers: int = 4,
    seed_worker: Callable[[int], None] = None,
    generator: torch.Generator = None,
) -> list[DataLoader]:
    dataloaders = []
    if ds_train:
        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        dataloaders.append(train_loader)
    if ds_val:
        val_loader = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        dataloaders.append(val_loader)
    if ds_test:
        test_loader = DataLoader(
            ds_test,
            batch_size=128,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        dataloaders.append(test_loader)
    return dataloaders


def hetero_dataloaders_from_hetero_data(
    hetero_data: HeteroData or Data,
    batch_size: int,
    num_neighbors: list[int],
    node_type: str,  # node type to use for neighbor sampling
    shuffle: bool = True,
    pin_memory: bool = True,
    num_workers: int = 4,
    generator: torch.Generator = None,
) -> tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
    train_loader = NeighborLoader(
        hetero_data,
        # Sample neighbors for each node and each edge type:
        num_neighbors=num_neighbors,
        # Use certain batch size for sampling training nodes of type "application":
        batch_size=batch_size,
        input_nodes=(node_type, hetero_data[node_type].train_mask),
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = NeighborLoader(
        hetero_data,
        # Sample neighbors for each node and each edge type:
        num_neighbors=num_neighbors,
        # Use certain batch size for sampling training nodes of type "application":
        batch_size=batch_size,
        input_nodes=(node_type, hetero_data[node_type].val_mask),
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        generator=generator,
    )
    test_loader = NeighborLoader(
        hetero_data,
        # Sample neighbors for each node and each edge type:
        num_neighbors=num_neighbors,
        # Use certain batch size for sampling training nodes of type "application":
        batch_size=batch_size,
        input_nodes=(node_type, hetero_data[node_type].test_mask),
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        generator=generator,
    )
    return train_loader, val_loader, test_loader
