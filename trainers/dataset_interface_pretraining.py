"""
Dataloader for pretraining datasets
"""

import os

import numpy as np
import torch
from tqdm import tqdm
import random


class DatasetInterface(torch.utils.data.IterableDataset):
    """
    Dataloaders for pretraining datasets, (ie. not with questions + answers).
    Returns: token ids
    """
    def __init__(self, split, cfg):
        """
        Arguments:
            cfg: the train script cfg
        """
        super().__init__()
        self.cfg = cfg
        self.context_window = self.cfg["model"]["context_window"]

        dataset_name = self.cfg["trainer"]["dataset"]["name"]
        self.data_path = os.path.join(
            self.cfg["general"]["paths"]["data_dir"],
            "data_as_memmaps",
            dataset_name,
            f"{split}.bin"
        )

        self._load_data()
        self.dataset_len = len(self.data) - self.context_window


    def _load_data(self):
        """
        Get data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} does not exist, preprocess the data first")
        self.data = np.memmap(
            self.data_path,
            dtype=np.uint16,
            mode="r",
        )
        print(f"Data loaded from {self.data_path}. Shape: {self.data.shape}")

    def __len__(self):
        """
        Return dataset length
        """
        return self.dataset_len
    
    def __iter__(self, idx):
        raise NotImplementedError
    
    
class BaseDatasetRandom(DatasetInterface):
    """
    Simple base dataloader for standard gpt-2'esk architectures and training.
    """
    def __init__(self, split, cfg):
        super().__init__(split, cfg)

    
    def __iter__(self):
        """
        Get a batch of random data points in an infinite loop.
        """
        while True:
            # Get a random index
            idx = random.randint(0, self.dataset_len - 1)
            
            # Extract a slice of data for x and y
            x = torch.from_numpy((self.data[idx: idx + self.context_window]).astype(np.int64))
            y = torch.from_numpy((self.data[idx + 1: idx + 1 + self.context_window]).astype(np.int64))
            attention_mask = None
            loss_mask = None

            # Yield the data points
            yield x, y



