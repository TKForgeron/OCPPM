import logging
import os
import pickle
from collections import defaultdict
from typing import Any

import pandas as pd
import torch
import torch_geometric
import torch_geometric.transforms as T
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm

from ..efg.efg import EFG

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(levelname)s %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     filename="logging/debug.log",
# )


class HOEG(EFG):
    """
    Class that serves as an adapter between ocpa and PyG.

    Specifically, it imports from a Feature_Storage class and works with PyG for implementing a GNN.

    TODO:
    - currently, we record y per node, maybe we should try saving one y per graph (only y for the last node in a graph)
    - add event indices as node indices, like in gnn_utils.py (generate_graph_dataset())
    - Add possibility to load Feature_Storage object from memory, instead of pickled file.
    """

    def __init__(
        self,
        root,
        events_filename: str,
        objects_filename: str,
        label_key: tuple[str, tuple],
        train: bool = False,
        validation: bool = False,
        test: bool = False,
        verbosity: int = 1,
        transform=None,
        pre_transform=None,
        file_extension: str = "pt",
        skip_cache: bool = False,
    ):
        """
        root (string, optional): Where the dataset should be stored. This folder is split
            into raw_dir (downloaded dataset) and processed_dir (processed data).

        train (bool, optional): If True, train indices of Feature_Storage will be used.
            Use this when constructing the train split of the data set.
            If train, validation, and test are all False, the whole Feature_Storage will
            be used as a data set (not recommended).

        validation (bool, optional): If True, validation indices of Feature_Storage will be used.
            Use this when constructing the validation split of the data set.
            If train, validation, and test are all False, the whole Feature_Storage will
            be used as a data set (not recommended).

        test (bool, optional): If True, test indices of Feature_Storage will be used.
            Use this when constructing the test split of the data set.
            If train, validation, and test are all False, the whole Feature_Storage will
            be used as a data set (not recommended).

        NOTE: For disambiguation purposes, througout this class, a distinction
            has been made between 'graph' and 'feature graph'.
            The first being of class `torch_geometric.data.Data` and the latter being
            of class `ocpa.algo.predictive_monitoring.obj.Feature_Graph.Feature_Storage`
        """
        self.data: FeatureStorage
        self.objects_data: dict[str, Any]
        self.events_filename = events_filename
        self.objects_filename = objects_filename
        self.label_key = label_key
        self.train = train
        self.validation = validation
        self.test = test
        self._verbosity = verbosity
        self._base_filename = "graph"
        if self.train:
            self._base_filename += "_train"
        elif self.validation:
            self._base_filename += "_val"
        elif self.test:
            self._base_filename += "_test"
        self._file_extension = file_extension
        self.skip_cache = skip_cache
        super(EFG, self).__init__(
            root, transform, pre_transform
        )  # use init of parent of EFG: Dataset

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        (The download function is not implemented here)
        """
        return [self.events_filename, self.objects_filename]

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""
        if self.skip_cache:
            # if (e.g. for testing purposes) we don't want to use cached files
            # we act as if we haven't found any processed file names
            return []
        else:
            # Retrieve Feature_Storage object from disk
            with open(self.raw_paths[0], "rb") as events_feature_storage:
                self.data = pickle.load(events_feature_storage)
            with open(self.raw_paths[1], "rb") as objects_data_dict:
                self.objects_data = pickle.load(objects_data_dict)

            if self.train:
                processed_file_names = [
                    self.__get_graph_filename(i)
                    for i in range(len(self.data.train_indices))
                ]
            elif self.validation:
                processed_file_names = [
                    self.__get_graph_filename(i)
                    for i in range(len(self.data.validation_indices))
                ]
            elif self.test:
                processed_file_names = [
                    self.__get_graph_filename(i)
                    for i in range(len(self.data.test_indices))
                ]
            else:
                processed_file_names = [
                    self.__get_graph_filename(i)
                    for i in range(len(self.data.feature_graphs))
                ]
            self.size = len(processed_file_names)
            return processed_file_names

    def process(self):
        """Processes a Feature_Storage object into PyG instance graph objects"""

        # Retrieve Feature_Storage object from disk
        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)
        with open(self.raw_paths[1], "rb") as objects_data_dict:
            self.objects_data = pickle.load(objects_data_dict)

        if self.train:
            # Retrieve feature graphs with train indices and write to disk
            self._feature_graphs_to_disk(
                [self.data.feature_graphs[i] for i in self.data.train_indices]
            )
        elif self.validation:
            # Retrieve graphs with validation indices and write to disk
            self._feature_graphs_to_disk(
                [self.data.feature_graphs[i] for i in self.data.validation_indices]
            )
        elif self.test:
            # Retrieve graphs with test indices and write to disk
            self._feature_graphs_to_disk(
                [self.data.feature_graphs[i] for i in self.data.test_indices]
            )
        else:
            # Write all graphs to disk
            self._feature_graphs_to_disk(self.data.feature_graphs)

    def _feature_graph_to_graph_to_disk(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        graph_idx: int,
    ) -> None:
        """Saves a FeatureStorage.Feature_Graph object as PyG Data object(s) to disk."""

        object_feature_vector_map, object_node_map = self._get_object_mapping(
            feature_graph=feature_graph,
            object_to_feature_vector_map=self.objects_data["object_feature_vector_map"],
        )

        # Split off labels from nodes,
        # and return full graph (cleansed of labels), and list of labels
        labels = self._split_X_y(feature_graph, self.label_key)
        # Get node features
        # TODO: currently only event nodes contain y-values,
        #       they could be added for object nodes in the future
        node_feats = self._get_node_features(
            feature_graph,
            self.objects_data["object_feature_matrices"],
            object_feature_vector_map,
        )
        # Get adjacency matrix per semantically sensible edge type
        edge_types = [
            ("event", "follows", "event"),
            ("event", "interacts", "application"),
            ("event", "interacts", "offer"),
        ]
        edge_index_dict = self._get_edge_index_dict(
            feature_graph=feature_graph,
            edge_types=edge_types,
            object_node_map=object_node_map,
        )

        # Define heterogeneous graph
        hetero_data = HeteroData()

        # Attach feature matrices and target variables
        hetero_data["event"].x = node_feats["event"]["x"]
        hetero_data["event"].y = labels
        hetero_data["application"].x = node_feats["application"]["x"]
        hetero_data["application"].y = node_feats["application"]["y"]
        hetero_data["offer"].x = node_feats["offer"]["x"]
        hetero_data["offer"].y = node_feats["offer"]["y"]

        # Define edge index per edge type
        for edge_type in edge_types:
            subject, predicate, object = edge_type[0], edge_type[1], edge_type[2]
            hetero_data[subject, predicate, object].edge_index = edge_index_dict[
                edge_type
            ]
        # application <-> application is not present in the dataset, so fill it with empty tensors
        hetero_data[
            "application", "interacts", "application"
        ].edge_index = torch.tensor([[], []], dtype=torch.int64)

        # Transform the HeteroData graph
        hetero_data = T.AddSelfLoops()(hetero_data)
        hetero_data = T.NormalizeFeatures()(hetero_data)

        torch.save(
            hetero_data,
            os.path.join(
                self.processed_dir,
                self.__get_graph_filename(graph_idx),
            ),
        )

    def _get_object_mapping(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        object_to_feature_vector_map: dict[str, dict[str, int]],
    ) -> tuple[dict[str, list[int]], dict[str, dict[str, int]]]:
        """
        Function that, given a feature_graph, returns a dictionary with a
        key per object type and indices as values. These indices are the
        rows of the object feature matrices (found in self.objects_data)
        that should be added as object nodes in the HeteroData graph.
        """

        # helper function:

        # get dict of with object type as keys and object ids as values
        unique_objects_dict = self.__set_to_split_dict(
            unique_objects=set(
                item for obj in feature_graph.objects.values() for item in obj
            )
        )
        # get dict of object type as keys and feature vector indices as values
        object_feature_vector_map = self.__replace_dict_values(
            unique_objects_dict, object_to_feature_vector_map
        )
        # get dict that maps object ids to node ids for each object type
        object_node_map = {
            key: {value: index for index, value in enumerate(value_list)}
            for key, value_list in unique_objects_dict.items()
        }

        return object_feature_vector_map, object_node_map

    def _split_X_y(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        label_key: tuple[str, tuple],
    ) -> torch.Tensor:
        """
        Impure function that splits off the target label from a feature graph
        and returns them both separately in a tuple of shape
        [A Feature_Graph, Number of Nodes]

        NOTE: This function should only be called once, since after it the label
        key is not present anymore, resulting in a KeyError.
        Also, it should be executed first in the processing pipeline (i.e. in self.process()).
        """
        ys = [node.attributes.pop(label_key) for node in feature_graph.nodes]

        return torch.tensor(ys, dtype=torch.float)

    def _get_node_features(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        object_feature_matrices: dict[str, pd.DataFrame],
        object_id_to_feature_matrix_index: dict[str, list[int]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        This will return a dict with  feature matrices per node type
        [Number of Nodes, Node Feature size]
        """
        # Append event node features to matrix
        event_node_feature_matrix: list[list[float]] = [
            list(node.attributes.values()) for node in feature_graph.nodes
        ]
        # Append application and offer node features to matrices,
        # selecting only the objects related to this execution graph/feature graph
        application_node_feature_matrix = (
            object_feature_matrices["application"]
            .drop(["application_index", "object_index"], axis=1)
            .iloc[object_id_to_feature_matrix_index["application"]]
        )
        offer_node_feature_matrix = (
            object_feature_matrices["offer"]
            .drop(["offer_index", "object_index"], axis=1)
            .iloc[object_id_to_feature_matrix_index["offer"]]
        )

        node_features_dict = {
            "event": {"x": torch.tensor(event_node_feature_matrix, dtype=torch.float)},
            "application": {
                "y": torch.tensor(
                    application_node_feature_matrix[
                        "@@object_lifecycle_duration"
                    ].values,
                    dtype=torch.float,
                ),
                "x": torch.tensor(
                    application_node_feature_matrix.drop(
                        columns=["@@object_lifecycle_duration"]
                    ).values,
                    dtype=torch.float,
                ),
            },
            "offer": {
                "y": torch.tensor(
                    offer_node_feature_matrix["@@object_lifecycle_duration"].values,
                    dtype=torch.float,
                ),
                "x": torch.tensor(
                    offer_node_feature_matrix.drop(
                        columns=["@@object_lifecycle_duration"]
                    ).values,
                    dtype=torch.float,
                ),
            },
        }
        return node_features_dict

    def _get_edge_index_dict(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        edge_types: list[tuple[str, str, str]],
        object_node_map: dict[str, dict[str, int]],
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """
        Returns the dictionary with directed adjacency matrices per node type, in COO format.
        """

        edge_index_dict = {}
        event_node_index_map = self.__get_event_node_index_mapping(feature_graph)

        for edge_type in edge_types:
            if edge_type == ("event", "follows", "event"):
                # Map event_id to node_index (counting from 0) using a dictionary
                # Actually map event_id to node_index
                # so we have an index-based (event_id-agnostic) directed COO adjacency_matrix.
                edge_index = [
                    (event_node_index_map[e.source], event_node_index_map[e.target])
                    for e in feature_graph.edges
                ]
            elif edge_type == ("event", "interacts", "application"):
                edge_index = self.__get_edge_index_for_edge_type(
                    feature_graph=feature_graph,
                    edge_type=edge_type,
                    event_node_map=event_node_index_map,
                    object_node_map=object_node_map["application"],
                )
            elif edge_type == ("event", "interacts", "offer"):
                edge_index = self.__get_edge_index_for_edge_type(
                    feature_graph=feature_graph,
                    edge_type=edge_type,
                    event_node_map=event_node_index_map,
                    object_node_map=object_node_map["offer"],
                )
            else:
                edge_index = "UNKNOWN EDGE TYPE GIVEN"
            edge_index_dict[edge_type] = torch.tensor(
                edge_index, dtype=torch.int64
            ).T  # ,dtype=torch.long)

        # use: self.objects_data['app_node_map'] for mapping oid to id in heterodata[ot]
        # use: event_node_index_map for mapping event_id to id in heterodata['event']

        return edge_index_dict

    def __set_to_split_dict(
        self, unique_objects: set[tuple[str, str]]
    ) -> dict[str, list[str]]:
        # Function that splits a set of [object type, object id]
        # into a dict with object types as keys and object ids as values
        result = defaultdict(list)
        for item in unique_objects:
            result[item[0]].append(item[1])
        return dict(result)

    def __replace_dict_values(
        self, split_dict: dict[str, list[str]], objects_map: dict[str, dict[str, int]]
    ) -> dict[str, list[int]]:
        return {
            key: [objects_map[key].get(value, value) for value in value_list]
            for key, value_list in split_dict.items()
        }

    def __get_edge_index_for_edge_type(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        edge_type: tuple[str, str, str],
        event_node_map: dict[int, int],
        object_node_map: dict[str, int],
    ) -> list[tuple[int, int]]:
        flatten = lambda nested_list: [
            item for sublist in nested_list for item in sublist
        ]

        # From all nodes in the feature graph:
        # get tuples that indicate which event_id interacts with which application/offer (oid from the OCEL)
        edge_list = [
            flatten(
                [
                    self.__get_event_object_edges(node, edge_type)
                    for node in feature_graph.nodes
                ]
            )
        ][0]
        # Map event_id to node_index for the application/offer node type
        edge_index = [
            (event_node_map[edge[0]], object_node_map[edge[1]]) for edge in edge_list
        ]
        return edge_index

    def __map_event_object_edge(
        self,
        event_id: int,
        oid: str,
        event_node_map: dict[int, int],
        object_node_map: dict[str, int],
    ) -> tuple[int, int]:
        return event_node_map[event_id], object_node_map[oid]

    def __get_event_object_edges(
        self,
        event_node: FeatureStorage.Feature_Graph.Node,
        edge_type: tuple[str, str, str],
    ) -> list[tuple[int, int]]:
        node_type = edge_type[2]
        edges = [
            (event_node.event_id, oid)
            for ot, oid in event_node.objects
            if ot == node_type
        ]
        return edges

    def __get_event_node_index_mapping(
        self, feature_graph: FeatureStorage.Feature_Graph
    ) -> dict[int, int]:
        """Returns a dictionary containing a mapping from event_ids to node indices in the given graph"""
        return {
            id: i
            for i, id in enumerate([node.event_id for node in feature_graph.nodes])
        }

    def __custom_verbosity_enumerate(self, iterable, miniters: int):
        """Returns either just the enumerated iterable, or one with the progress tracked."""
        if self._verbosity:
            return tqdm(enumerate(iterable), miniters=miniters)
        else:
            return enumerate(iterable)

    def __get_graph_filename(self, graph_idx: int) -> str:
        return f"{self._base_filename}_{graph_idx}.{self._file_extension}"

    def len(self) -> int:
        return self.size

    def get(self, graph_idx):
        """
        - Equivalent to __getitem__ in PyTorch
        - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(
            os.path.join(self.processed_dir, self.__get_graph_filename(graph_idx))
        )
        return data
