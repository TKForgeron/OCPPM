# Python native
from typing import Any, Callable, Union

# PyG
import torch
# PyTorch TensorBoard support
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Custom imports
from loan_application_experiment.feature_encodings.efg.efg import EFG
from loan_application_experiment.feature_encodings.efg.efg_sg import EFG_SG
from models.definitions.geometric_models import GraphModel


def load_datasets(
    dataset_class: Union[EFG, EFG_SG],
    storage_path: str,
    split_feature_storage_file: str,
    target_label: tuple[str, tuple],
    graph_level_target: bool,
    features_dtype: torch.dtype,
    target_dtype: torch.dtype,
    subgraph_size: int,
    transform=None,
    train: bool = None,
    val: bool = None,
    test: bool = None,
    skip_cache: bool = False,
) -> Union[list[EFG],list[EFG_SG]]:
    kwargs = {
        "features_dtype": features_dtype,
        "target_dtype": target_dtype,
        "size_subgraph_samples": subgraph_size,
        "transform": transform,
        "verbosity": 51,
        "skip_cache": skip_cache,
    }
    if dataset_class == EFG:
        del kwargs["size_subgraph_samples"]

    datasets = []
    if train:
        ds_train = dataset_class(
            train=True,
            root=storage_path,
            filename=split_feature_storage_file,
            label_key=target_label,
            graph_level_target=graph_level_target,
            **kwargs
        )
        datasets.append(ds_train)
    if val:
        ds_val = dataset_class(
            validation=True,
            root=storage_path,
            filename=split_feature_storage_file,
            label_key=target_label,
            graph_level_target=graph_level_target,
            **kwargs
        )
        datasets.append(ds_val)
    if test:
        ds_test = dataset_class(
            test=True,
            root=storage_path,
            filename=split_feature_storage_file,
            label_key=target_label,
            graph_level_target=graph_level_target,
            **kwargs
        )
        datasets.append(ds_test)
    return datasets


def print_dataset_summaries(
    ds_train: EFG_SG or EFG = None,
    ds_val: EFG_SG or EFG = None,
    ds_test: EFG_SG or EFG = None,
) -> None:
    if ds_train:
        print("Train set")
        print(ds_train.get_summary(), "\n")
    if ds_val:
        print("Validation set")
        print(ds_val.get_summary(), "\n")
    if ds_test:
        print("Test set")
        print(ds_test.get_summary(), "\n")


def prepare_dataloaders(
    batch_size: int,
    ds_train: EFG_SG or EFG = None,
    ds_val: EFG_SG or EFG = None,
    ds_test: EFG_SG or EFG = None,
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
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        dataloaders.append(test_loader)
    return dataloaders
