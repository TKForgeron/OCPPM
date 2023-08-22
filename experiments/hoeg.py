# import logging
import os
import pickle
from collections import defaultdict
from typing import Any, Union

import pandas as pd
import torch
# import torch_geometric
# import torch_geometric.transforms as T
from ocpa.algo.predictive_monitoring.obj import \
    Feature_Storage as FeatureStorage
from torch_geometric.data import Dataset, HeteroData

from experiments.efg import EFG

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(levelname)s %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     filename="logging/DEBUG.log",
# )
# logging.debug("-" * 32 + " hoeg.py " + "-" * 32)


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
        event_node_label_key: Union[str, tuple[str, tuple]],
        object_nodes_label_key: str,
        edge_types: list[tuple[str, str, str]],
        object_node_types: list[str],
        event_node_type: str = "event",
        graph_level_target: bool = False,
        target_dtype: torch.dtype = torch.float32,
        features_dtype: torch.dtype = torch.float32,
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
        self.event_node_label_key = event_node_label_key
        self.object_nodes_label_key = object_nodes_label_key
        self.event_node_type = event_node_type
        self.object_node_types = object_node_types
        self.edge_types = edge_types
        self.graph_level_target = graph_level_target
        self.target_dtype = target_dtype
        self.features_dtype = features_dtype
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
            del self.data
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
        graph_level_target: bool,
    ) -> None:
        """Saves a FeatureStorage.Feature_Graph object as PyG Data object(s) to disk."""
        self.DEBUG_fg_idx = feature_graph.pexec_id
        # Get object node information with respect to current process execution graph
        object_feature_vector_map, object_node_map = self._get_object_mapping(
            feature_graph=feature_graph,
            object_to_feature_vector_map=self.objects_data["object_feature_vector_map"],
        )
        # Split off labels from nodes, and return dict with tensor of event node labels
        event_node_labels = self._split_X_y(
            feature_graph=feature_graph,
            event_node_type=self.event_node_type,
            label_key=self.event_node_label_key,
        )
        # Get node features
        node_feats = self._get_node_features(
            event_node_type=self.event_node_type,
            feature_graph=feature_graph,
            object_node_types=self.object_node_types,
            object_feature_matrices=self.objects_data["object_feature_matrices"],
            object_id_to_feature_matrix_index=object_feature_vector_map,
            object_nodes_label_key=self.object_nodes_label_key,
        )
        # Get adjacency matrix per semantically sensible edge type
        edge_index_dict = self._get_hetero_edge_index_dict(
            event_node_type=self.event_node_type,
            feature_graph=feature_graph,
            edge_types=self.edge_types,
            object_node_map=object_node_map,
        )

        # Define heterogeneous graph
        hetero_data = self._get_hetero_graph_object(
            event_node_type=self.event_node_type,
            object_node_types=self.object_node_types,
            edge_types=self.edge_types,
            node_features=node_feats,
            edge_index_dict=edge_index_dict,
            event_node_labels=event_node_labels,
        )

        torch.save(
            hetero_data,
            os.path.join(
                self.processed_dir,
                self.__get_graph_filename(graph_idx),
            ),
        )

    def _get_hetero_graph_object(
        self,
        event_node_type: str,
        object_node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        node_features: dict[str, dict[str, torch.Tensor]],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        event_node_labels: dict[str, torch.Tensor],
    ) -> HeteroData:
        # Initialize PyTorch Geometric's HeteroData object
        hetero_data = HeteroData()
        # Attach feature matrices and target variables
        hetero_data[event_node_type].x = node_features[event_node_type]["x"]
        hetero_data[event_node_type].y = event_node_labels[event_node_type]
        for object_node_type in object_node_types:
            if object_node_type not in node_features:
                # continue # if object type not related to current process execution graph, skip loop and try next object type
                hetero_data[object_node_type].x = torch.tensor(
                    [], dtype=self.features_dtype
                )
                hetero_data[object_node_type].y = torch.tensor(
                    [], dtype=self.target_dtype
                )
            else:
                hetero_data[object_node_type].x = node_features[object_node_type]["x"]
                hetero_data[object_node_type].y = node_features[object_node_type]["y"]
        # Define edge index per edge type
        for edge_type in edge_types:
            subject, predicate, object = edge_type[0], edge_type[1], edge_type[2]
            if edge_type in edge_index_dict:
                edge_index = edge_index_dict[edge_type]
            else:
                edge_index = torch.tensor([[], []], dtype=torch.int64)
            hetero_data[subject, predicate, object].edge_index = edge_index
        # TODO: test whether non existing edge_types should be defined empty, or just not defined
        #       I think they should be defined so PyG can add self-loops
        #       (but then again, do we want that for object_node_types?)
        # # application <-> application is not present in the dataset, so fill it with empty tensors
        # hetero_data[
        #     "application", "interacts", "application"
        # ].edge_index = torch.tensor([[], []], dtype=torch.int64)

        return hetero_data

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

        # get dict with object type as keys and object ids as values
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
        event_node_type: str,
        label_key: Union[str, tuple[str, tuple]],
    ) -> dict[str, torch.Tensor]:
        """
        Impure function that splits off the target label from a feature graph
        and returns them both separately in a tuple of shape
        [A Feature_Graph, Number of Nodes]

        NOTE: This function should only be called once, since after it the label
        key is not present anymore, resulting in a KeyError.
        Also, it should be executed first in the processing pipeline (i.e. in self.process()).
        """
        ys = [node.attributes.pop(label_key) for node in feature_graph.nodes]

        return {event_node_type: torch.tensor(ys, dtype=self.target_dtype)}

    def _get_node_features(
        self,
        event_node_type: str,
        feature_graph: FeatureStorage.Feature_Graph,
        object_node_types: list[str],
        object_feature_matrices: dict[str, pd.DataFrame],
        object_id_to_feature_matrix_index: dict[str, list[int]],
        object_nodes_label_key: str,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        This will return a dict with  feature matrices per node type
        [Number of Nodes, Node Feature size]
        """
        # Append event node features to matrix
        event_node_feature_matrix: list[list[float]] = [
            list(node.attributes.values()) for node in feature_graph.nodes
        ]

        node_features_dict = {
            event_node_type: {
                "x": torch.tensor(event_node_feature_matrix, dtype=self.features_dtype)
            }
        }
        for object_node_type in object_node_types:
            if object_node_type in object_id_to_feature_matrix_index:
                if type(object_id_to_feature_matrix_index[object_node_type][0]) != int:
                    # logging.debug(f"in self._get_node_features()")
                    # logging.debug(f"fg.pexec_id: {self.DEBUG_fg_idx}")
                    # logging.debug(object_id_to_feature_matrix_index[object_node_type])
                    continue  # cs BUG: somehow 148 objects are prepared incorrectly in `object_id_to_feature_matrix_index`
                object_node_feature_matrix = (
                    object_feature_matrices[object_node_type]
                    .drop(
                        [f"{object_node_type}_index", "object_index"], axis=1
                    )  # assuming naming scheme in ofg construction pipeline
                    .iloc[object_id_to_feature_matrix_index[object_node_type]]
                )
                node_features_dict[object_node_type] = {
                    "y": torch.tensor(
                        object_node_feature_matrix[object_nodes_label_key].values,
                        dtype=self.target_dtype,
                    ),
                    "x": torch.tensor(
                        object_node_feature_matrix.drop(
                            columns=[object_nodes_label_key]
                        ).values,
                        dtype=self.features_dtype,
                    ),
                }
        return node_features_dict

    def _get_hetero_edge_index_dict(
        self,
        event_node_type: str,
        feature_graph: FeatureStorage.Feature_Graph,
        edge_types: list[tuple[str, str, str]],
        object_node_map: dict[str, dict[str, int]],
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """
        Returns the dictionary with directed adjacency matrices per node type, in COO format.
        """
        edge_index_dict = {}
        for edge_type in edge_types:
            if not event_node_type in edge_type:
                raise ValueError(
                    f"UNKNOWN EDGE TYPE GIVEN, at least one of the node types in the given edge type should be '{event_node_type}' (we don't directly connect objects with objects)"
                )
            else:
                event_node_index_map = self.__get_event_node_index_mapping(
                    feature_graph
                )
                if edge_type.count(event_node_type) == 2:
                    # Map event_id to node_index (counting from 0) using a dictionary
                    # Actually map event_id to node_index
                    # so we have an index-based (event_id-agnostic) directed COO adjacency_matrix.
                    edge_index = [
                        (event_node_index_map[e.source], event_node_index_map[e.target])
                        for e in feature_graph.edges
                    ]
                else:
                    object_node_type = edge_type[
                        edge_type[::-1].index(event_node_type)
                    ]  # determine which object node type this edge concerns
                    if not object_node_type in object_node_map:
                        continue  # if object type not related to current process execution graph, skip loop and try next object type
                    edge_index = self.__get_edge_index_for_edge_type(
                        feature_graph=feature_graph,
                        edge_type=edge_type,
                        event_node_map=event_node_index_map,
                        object_node_map=object_node_map[object_node_type],
                    )
            edge_index_dict[edge_type] = torch.tensor(edge_index, dtype=torch.int64).T
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
        edge_index = []
        for edge in edge_list:
            if edge[1] in object_node_map:
                edge_index.append((event_node_map[edge[0]], object_node_map[edge[1]]))
            else:
                # logging.debug(f"in self.__get_edge_index_for_edge_type()")
                # logging.debug(f"fg.pexec_id: {self.DEBUG_fg_idx}")
                # logging.debug(edge)
                # logging.debug(object_node_map)
                edge_index.append(
                    (event_node_map[edge[0]], 0)
                )  # cs BUG: sometimes there is a mismatch between edge[1] and keys in `object_node_map`
        # edge_index = [
        #     (event_node_map[edge[0]], object_node_map[edge[1]]) for edge in edge_list
        # ]
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

    def __get_graph_filename(self, graph_idx: int) -> str:
        filename = f"{self._base_filename}_{graph_idx}.{self._file_extension}"
        # logging.debug(filename)
        return filename

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
        # logging.debug(data)
        return data
