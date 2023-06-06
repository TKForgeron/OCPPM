# import pandas as pd
import os
import pickle

import torch
import torch_geometric
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")
print()


class EFG(Dataset):
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
        filename,
        label_key: tuple[str, tuple],
        train: bool = False,
        validation: bool = False,
        test: bool = False,
        verbosity: int = 1,
        transform=None,
        pre_transform=None,
        file_extension: str = "pt",
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
        self.filename = filename
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
        super(EFG, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""

        # Retrieve Feature_Storage object from disk
        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)

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
                self.__get_graph_filename(i) for i in range(len(self.data.test_indices))
            ]
        else:
            processed_file_names = [
                self.__get_graph_filename(i)
                for i in range(len(self.data.feature_graphs))
            ]
        self.size = len(processed_file_names)
        return processed_file_names

    def _set_size(self, size: int) -> None:
        """Sets the number of graphs stored in this EventGraphDataset object."""
        self._size = size

    def _get_size(self) -> int:
        """Gets the number of graphs stored in this EventGraphDataset object."""
        return self._size

    size: int = property(_get_size, _set_size)

    def process(self):
        """Processes a Feature_Storage object into PyG instance graph objects"""

        # Retrieve Feature_Storage object from disk
        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)

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

    def _feature_graphs_to_disk(
        self,
        feature_graphs: list[FeatureStorage.Feature_Graph],
    ):
        # Set size of the dataset
        self.size = len(feature_graphs)
        # Save each feature_graph instance
        for index, feature_graph in self.__custom_verbosity_enumerate(
            feature_graphs, miniters=self._verbosity
        ):
            # Save a feature_graph instance
            self._feature_graph_to_graph_to_disk(
                feature_graph=feature_graph,
                graph_idx=index,
            )

    def _feature_graph_to_graph_to_disk(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        graph_idx: int,
    ) -> None:
        """Saves a FeatureStorage.Feature_Graph object as PyG Data object(s) to disk."""
        # Split off labels from nodes,
        # and return full graph (cleansed of labels), and list of labels
        labels = self._split_X_y(feature_graph, self.label_key)
        # Get node features
        node_feats = self._get_node_features(feature_graph)
        # Get edge features
        # edge_feats = self._get_edge_features(feature_graph)
        # Get adjacency matrix
        edge_index = self._get_adjacency_matrix(feature_graph)
        # Create graph data object
        data = Data(
            y=labels,
            x=node_feats,
            edge_index=edge_index,
            # edge_attr=edge_feats,
        )

        torch.save(
            data,
            os.path.join(
                self.processed_dir,
                self.__get_graph_filename(graph_idx),
            ),
        )

    def _split_X_y(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        label_key: tuple[str, tuple],
    ) -> list[torch.float]:
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
        self, feature_graph: FeatureStorage.Feature_Graph
    ) -> torch.Tensor:
        """
        This will return a feature matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        # Append node features to matrix
        node_feature_matrix: list[list[torch.float]] = [
            list(node.attributes.values()) for node in feature_graph.nodes
        ]

        return torch.tensor(node_feature_matrix, dtype=torch.float)

    def _get_edge_features(self, feature_graph: FeatureStorage.Feature_Graph):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        pass

    def _get_adjacency_matrix(
        self, feature_graph: FeatureStorage.Feature_Graph
    ) -> torch.Tensor:
        """
        Returns the directed adjacency matrix in COO format, given a graph
        [2, Number of edges]
        """
        # Map event_id to node_index (counting from 0) using a dictionary
        node_index_map = self.__get_node_index_mapping(feature_graph)
        # Actually map event_id to node_index
        # so we have an index-based (event_id-agnostic) directed COO adjacency_matrix.
        adjacency_matrix_COO = [
            [node_index_map[e.source] for e in feature_graph.edges],
            [node_index_map[e.target] for e in feature_graph.edges],
        ]

        return torch.tensor(adjacency_matrix_COO, dtype=torch.long)

    def __get_node_index_mapping(
        self, feature_graph: FeatureStorage.Feature_Graph
    ) -> dict:
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
