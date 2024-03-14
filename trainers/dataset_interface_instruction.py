"""
Dataloader for instruction datasets
"""

import os
import json

import numpy as np
import torch
from tqdm import tqdm
import random
from dataclasses import dataclass


class DatasetInterfaceInstruction(torch.utils.data.IterableDataset):
    """
    Dataloaders for instruction datasets, ie. with a question and answer.
    Concatenates multiple question-answer examples together to fill the context window.
    Returns:
        token ids
        attention mask (for the question-answer examples)
        loss mask (for the answer)
    """
    def __init__(self, split: str, cfg):
        """
        Arguments:
            cfg: the train script cfg
        """
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.context_window = self.cfg["model"]["context_window"]

        data_path = os.path.join(
            cfg["general"]["paths"]["data_dir"],
            "data_as_memmaps",
            cfg["trainer"]["dataset"]["name"],
        )
        self.data, self.idxs_for_masks, self.dataset_len = self._load_data(split, data_path)
        self.max_seq_length = self.data.shape[1]
        assert self.max_seq_length <= self.context_window, f"{self.max_seq_length=} {self.context_window=}"
        

    @staticmethod
    def _load_data(split, as_memmaps_path):
        """
        Get data
        """
        ### Load the shapes
        shapes_path = os.path.join(as_memmaps_path, f"shapes.json")
        if not os.path.exists(shapes_path):
            raise FileNotFoundError(f"{shapes_path} does not exist, preprocess the data first")
        with open(shapes_path, "r") as f:
            shapes = json.load(f)
        dataset_len = int(shapes[f"{split}_size"])

        ### Paths
        data_path = os.path.join(as_memmaps_path, f"{split}_data.bin")
        idxs_for_masks_path = os.path.join(as_memmaps_path, f"{split}_idxs_for_masks.bin")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} does not exist, preprocess the data first")
        if not os.path.exists(idxs_for_masks_path):
            raise FileNotFoundError(f"{idxs_for_masks_path} does not exist, preprocess the data first")

        ### Load the data, and idxs_for_masks
        data = np.memmap(
            data_path,
            dtype=np.uint16,
            mode="r",).reshape(shapes[f"{split}_size"], -1)
        idxs_for_masks = np.memmap(
            idxs_for_masks_path,
            dtype=np.uint16,
            mode="r",).reshape(shapes[f"{split}_size"], 3)
        
        print(f"-----------------------------------")
        print(f"Data loaded to DatasetInterfaceInstruction:")
        print(f"Split: {split}")
        print(f"Path: {data_path}")
        print(f"Data shape: {data.shape}")
        print(f"idxs_for_masks shape: {idxs_for_masks.shape}")
        print(f"average_seq_length: {np.mean(idxs_for_masks[:, -1])}")
        print(f"min_seq_length: {np.min(idxs_for_masks[:, -1])}")
        print(f"max_seq_length: {np.max(idxs_for_masks[:, -1])}")
        print(f"-----------------------------------")

        return data, idxs_for_masks, dataset_len
        

    def __len__(self):
        """
        Return dataset length
        """
        return self.dataset_len

    def __iter__(self):
        """
        Samples question-answer examples and concatenates them together
        Returns the concatenated examples and the attention mask and loss mask.
        """
        while True:
            token_ids = []
            idxs_for_masks = []
            length = 0
            while length < self.context_window+1:
                i =  np.random.randint(0, self.dataset_len - 1)
                end = self.idxs_for_masks[i][-1]
                token_ids.append(torch.from_numpy(self.data[i][:end].astype(np.int64)))
                idxs_for_masks.append(torch.from_numpy(self.idxs_for_masks[i].astype(np.int64))+length)
                length += end

            # Concatenate examples together and truncate
            token_ids = torch.cat(token_ids)
            x = token_ids[:self.context_window]
            idxs_for_masks = torch.stack(idxs_for_masks) # shape=(N, D) where N = num examples, D = num idxs for masks
            ### Check idxs_for_masks are concatenated correctly. End index of one example should be the start index of the next.
            for i in range(1, len(idxs_for_masks)):
                assert idxs_for_masks[i][0] == idxs_for_masks[i-1][-1], "Examples are not concatenated correctly"

            attention_mask = build__attention_mask(idxs_for_masks, token_ids)
            attention_mask = attention_mask[:self.context_window, :self.context_window]
            loss_mask = build__loss_mask(idxs_for_masks, token_ids)

            y = token_ids[1:self.context_window+1]
            loss_mask = loss_mask[1:self.context_window+1]

            # Yield the data points
            yield x, y, attention_mask, loss_mask


def build__attention_mask(idxs_for_masks, x):
    """Build attention mask for a single example."""
    context_window = x.shape[0]
    # start with a standard causal attention mask
    attn_mask = torch.ones(context_window, context_window, dtype=torch.bool, device=x.device
                           ).tril(diagonal=0)
    for idxs_for_masks_ in idxs_for_masks[1:]:
        # mask out the tokens before the start of the current example
        start, end = idxs_for_masks_[0], idxs_for_masks_[-1]
        # attn_mask[start:end, , :start] = False
        attn_mask[start:min(end, context_window), :start] = False
    return attn_mask

def build__loss_mask(idxs_for_masks, x):
    """Build loss mask for a single example."""
    context_window = x.shape[0]
    num_examples, d = idxs_for_masks.shape
    loss_mask = torch.zeros(context_window, dtype=torch.bool)
    for idxs_for_masks_ in idxs_for_masks:
        for i in range(1, d, 2):
            loss_mask[min(idxs_for_masks_[i], context_window):min(idxs_for_masks_[i+1], context_window)] = True
    return loss_mask