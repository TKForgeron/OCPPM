# import pandas as pd
import os
import pickle

import torch
import torch_geometric
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from experiment.feature_encodings.efg.efg import EFG

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")
print()


class CEEG(EFG):
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
        super().__init__(
            root,
            filename,
            label_key,
            train,
            validation,
            test,
            verbosity,
            transform,
            pre_transform,
            file_extension,
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
            #
            # implement something to add graph-level features
            # (they are already present in FeatureStorage)
            #
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
