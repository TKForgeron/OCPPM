import logging
import os
import os.path as osp
import pickle
from typing import Callable, Optional

import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset


class BPI17_OFG(InMemoryDataset):
    """
    Args:
        root (str): Root directory where the dataset should be saved.
            Try: "data/BPI17/feature_encodings"
        preprocess (str, optional): Pre-processes the original dataset by
            adding structural features (:obj:`"metapath2vec"`, :obj:`"TransE"`)
            to featureless nodes. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(
        self,
        root: str,  # "data/BPI17/feature_encodings"
        preprocess: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.filename = "BPI17_OFG"
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "ofg", "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "ofg", "processed")

    @property
    def raw_file_names(self) -> list[str]:
        return [self.filename]

    @property
    def processed_file_names(self) -> str:
        if self.preprocess is not None:
            return f"data_{self.preprocess}.pt"
        else:
            return "data.pt"

    def process(self):
        print(os.getcwd())
        with open(f"{self.raw_paths[0]}.pkl", "rb") as file:
            # self.data = pickle.load(file)
            hetero_data = pickle.load(file)
        hetero_data = T.AddSelfLoops()(hetero_data)
        hetero_data = T.NormalizeFeatures()(hetero_data)
        hetero_data = T.RandomNodeSplit()(hetero_data)

        logging.debug("hetero_data:", hetero_data)
        # with open(f"{self.root}/{self.filename}.pkl", "rb") as binary_file:
        #     hetero_data = pickle.load(binary_file)

        torch.save(self.collate([hetero_data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return "BPI17-OFG()"
