# Python native
import os

os.chdir("/home/tim/Development/OCPPM/")

# %%
# import logging
from typing import Any, Callable, Union

import pandas as pd
# PyG
import torch
# PyTorch TensorBoard support
import torch.utils.tensorboard
import torch_geometric.transforms.to_undirected as T
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader

# Custom imports
from experiments.hoeg import HOEG

# %%


class ToUndirected(T.BaseTransform):
    """
    This is a custom tranform that can be applied to
    PyG's Data and HeteroData objects to make them
    undirected graphs.

    It is different from PyG's ToUndirected in that
    here we can specify edge types to exclude from
    tranformations.
    """

    def __init__(
        self,
        exclude_edge_types: list[tuple[str, str, str]]=[],
        reduce: str = "add",
        merge: bool = True,
    ):
        self.exclude_edge_types = exclude_edge_types
        self.reduce = reduce
        self.merge = merge

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store._key in self.exclude_edge_types:
                continue
            if "edge_index" not in store:
                continue

            nnz = store.edge_index.size(1)

            if isinstance(data, HeteroData) and (
                store.is_bipartite() or not self.merge
            ):
                src, rel, dst = store._key

                # Just reverse the connectivity and add edge attributes:
                row, col = store.edge_index
                rev_edge_index = torch.stack([col, row], dim=0)

                inv_store = data[dst, f"rev_{rel}", src]
                inv_store.edge_index = rev_edge_index
                for key, value in store.items():
                    if key == "edge_index":
                        continue
                    if isinstance(value, torch.Tensor) and value.size(0) == nnz:
                        inv_store[key] = value

            else:
                keys, values = [], []
                for key, value in store.items():
                    if key == "edge_index":
                        continue

                    if store.is_edge_attr(key):
                        keys.append(key)
                        values.append(value)

                store.edge_index, values = T.to_undirected(
                    store.edge_index, values, reduce=self.reduce
                )

                for key, value in zip(keys, values):
                    store[key] = value

        return data

class FillEmptyNodeTypes(T.BaseTransform):
    def __init__(
        self,
        fill_value: float = 1,
        fill_value_dtype=torch.float,
    ):
        self.fill_value = fill_value
        self.fill_value_dtype = fill_value_dtype

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items('x'):
                # fill empty data store with `self.fill_value`
                if value.shape[0] == 0:
                    store[key] = torch.tensor([[self.fill_value]],dtype=self.fill_value_dtype)
        return data
    
class NormalizeFeatures(T.BaseTransform):
    def __init__(self, attrs: list[str] = ["x"]):
        self.attrs = attrs

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.shape[0] != 0:
                    value = value - value.min()
                    value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
                    store[key] = value
        return data



def edge_types_to_undirected(edge_types: list[tuple[str, str]]):
    undirected_edges = list(edge_types)
    for edge in edge_types:
        reversed_edge = (edge[1], edge[0])
        if reversed_edge not in undirected_edges:
            undirected_edges.append(reversed_edge)
    return undirected_edges


def get_index_map(df: pd.DataFrame, index_col_a: str, index_col_b: str) -> dict:
    return pd.Series(df[index_col_b].values, index=df[index_col_a]).to_dict()


def add_object_type_index(df: pd.DataFrame, object_type: str) -> pd.DataFrame:
    return (
        df.reset_index()
        .drop("index", axis=1)
        .reset_index()
        .rename(columns={"index": f"{object_type}_index"})
    )


def object_map_to_node_map(object_map: dict, node_map: dict, object_name: str) -> dict:
    return {
        k: node_map[v]
        for (k, v) in object_map.items()
        if object_name.lower() in k.lower()
    }


def split_on_edge_types(
    edge_list: list[tuple[str, str]],
    edge_types: list[tuple[str, str]],
    to_undirected: bool = False,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """
    Function that splits edges based on a given list of edge types.
    It returns a dict, with a key for each (found) edge type and a list for the corresponding edges.

    Currently, the function assumes:
        - Directed edges. So do specify both ways, if undirected edges are wanted.
        - Object type names are in the object IDs (found in edge_list).

    Procedure:
    Per edge type (in edge_types), append all edges (from edge_list)
    if both the source and destination type of the edge corresponds to the edge type's source and destination
    """
    edge_types = edge_types_to_undirected(edge_types) if to_undirected else edge_types

    edges_split_on_type = dict()
    for edge_type in edge_types:
        for edge in edge_list:
            if edge_type[0] in edge[0].lower() and edge_type[1] in edge[1].lower():
                if edge_type in edges_split_on_type:
                    # if the key/edge_type already exists, append
                    edges_split_on_type[edge_type].append(edge)
                else:
                    # if this edge is the first of this edge_type
                    edges_split_on_type[edge_type] = [
                        edge
                    ]  # define a new key and value (list with the edge inside)
    return edges_split_on_type


def to_undirected(edge_list: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    return list(set(edge_list + [tuple(reversed(edge)) for edge in edge_list]))


def rename_edges(
    edges: list[tuple[Any, Any]],
    renaming_map: dict,
) -> list[tuple[Any, Any]]:
    renamed_edges = []
    for edge in edges:
        if edge[0] in renaming_map and edge[1] in renaming_map:
            renamed_edges.append((renaming_map[edge[0]], renaming_map[edge[1]]))
        elif edge[0] in renaming_map:
            renamed_edges.append((renaming_map[edge[0]], edge[1]))
        elif edge[1] in renaming_map:
            renamed_edges.append((edge[0], renaming_map[edge[1]]))
        else:
            renamed_edges.append(edge)
    return renamed_edges


def rename_edges_in_split_dict(
    split_edges: dict[tuple[str, str], list[tuple[Any, Any]]], renaming_map: dict
) -> dict[tuple[str, str], list[tuple[Any, Any]]]:
    return {
        edge_type: rename_edges(edges, renaming_map)
        for (edge_type, edges) in split_edges.items()
    }


def to_torch_coo_format(edge_list: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edge_list, dtype=torch.int64).T


def load_hetero_datasets(
    storage_path: str,
    split_feature_storage_file: str,
    objects_data_file: str,
    event_node_label_key: Union[str, tuple[str, tuple]],
    object_nodes_label_key: str,
    edge_types: list[tuple[str, str, str]],
    object_node_types: list[str],
    event_node_type: str = "event",
    graph_level_target: bool = False,
    pre_transform=None,
    transform=None,
    train: bool = True,
    val: bool = True,
    test: bool = True,
    verbosity: int = 51,
    skip_cache: bool = False,
) -> list[HOEG]:
    datasets = []
    kwargs = {
        "root": storage_path,
        "events_filename": split_feature_storage_file,
        "objects_filename": objects_data_file,
        "event_node_label_key": event_node_label_key,
        "object_nodes_label_key": object_nodes_label_key,
        "edge_types": edge_types,
        "object_node_types": object_node_types,
        "event_node_type": event_node_type,
        "graph_level_target": graph_level_target,
        'pre_transform':pre_transform,
        "transform": transform,
        "verbosity": int(verbosity),
        "skip_cache": skip_cache,
    }

    if train:
        ds_train = HOEG(train=True, **kwargs)
        datasets.append(ds_train)
    if val:
        ds_val = HOEG(validation=True, **kwargs)
        datasets.append(ds_val)
    if test:
        ds_test = HOEG(test=True, **kwargs)
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
            batch_size=batch_size,
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
