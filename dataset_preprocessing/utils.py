"""
Necessary to be run before training to make sure all of the data is preprcessed etc.
"""
import os 
import numpy as np 
from tqdm import tqdm

def write_tokenized_data_as_memmap(tokenized, tokenized_data_folder):
    """
    Write the tokenized data to a file
    """
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(tokenized_data_folder, f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(
            filename, 
            dtype=dtype, 
            mode="w+", 
            shape=(arr_len,)
        )
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    

def save_as_memmaps(data, name, as_memmaps_path):
    arr = np.memmap(os.path.join(as_memmaps_path, f"{name}.bin"), dtype=np.uint16, mode="w+", shape=data.shape)
    arr[:] = data
    arr.flush()
