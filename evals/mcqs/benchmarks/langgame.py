import random
from datasets import DatasetDict
import os
def load_langgame(split, **kwargs):
    """Load and process the benchmark"""
    current_file_path = os.path.abspath(__file__)
    dir_of_file = os.path.dirname(current_file_path)
    as_datasets_path = os.path.join(dir_of_file, "../../../data/data_as_datasets/langgame")
    base_dataset = DatasetDict.load_from_disk(as_datasets_path)[split]
    print(f"LangGame: Loaded {len(base_dataset)} examples from {split} split.")
    index = list(range(len(base_dataset)))
    random.shuffle(index)
    for i in index:
        sample = base_dataset[i]
        prefix = sample["question"]
        options = sample["options"]
        ground_truth = options[0]
        false_options = options[1:]
        yield prefix, ground_truth, false_options
