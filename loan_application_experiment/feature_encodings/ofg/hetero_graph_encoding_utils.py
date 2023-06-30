# %%
# import logging
from typing import Any

import pandas as pd
import torch


# %%
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
    edge_list: list[tuple[str, str]], edge_types: list[tuple[str, str]]
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """
    Function that splits edges based on a given list of edge types.
    It returns a dict, with a key for each (found) edge type and a list for the corresponding edges.

    Currently, the function assumes directed edges. So do specify both ways, if undirected edges are wanted.

    Procedure:
    Per edge type (in edge_types), append all edges (from edge_list)
    if both the source and destination type of the edge corresponds to the edge type's source and destination
    """

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
