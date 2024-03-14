"""
A simple wrapper around the GPT2 Tokenizer to
standardize the interface for tokenization.
"""

import os
import json
from tqdm import tqdm
import random
import numpy as np

class StochastokProcessor:
    """
    A processor that applies stochastok expansion to the tokenized data.
    """
    def __init__(self, tokenizer, expand_prop=None):
        self.tokenizer = tokenizer
        self.expand_prop = expand_prop
        self.set_expansions()

    def expand(self, token_ids, expand_prop=0.1, max_num_to_expand=None, disable_tqdm=True):
        """
        Expand the sequence of tokens by splitting tokens.
        Args:
            token_ids (list): list of token_ids to expand.
            expand_prop (float): proportion of tokens to (try) expanding.
        Returns:
            list: list of token_ids after expanding.
        """
        expand_prop = expand_prop if expand_prop is not None else self.expand_prop
        num_to_expand = int(len(token_ids) * expand_prop)
        num_expanded = 0
        for _ in tqdm(range(num_to_expand), disable=disable_tqdm):
            if max_num_to_expand is not None and num_expanded >= max_num_to_expand:
                break
            idx = np.random.randint(len(token_ids))
            token_id = token_ids[idx]
            if token_id in self.expansions:
                possible_expansions = self.expansions[token_id]
                chosen_expansion = random.choice(possible_expansions)
                token_ids = token_ids[:idx] + list(chosen_expansion) + token_ids[idx+1:]
                num_expanded += 1
        # print(f"Attempted to expand {num_to_expand}, expanded {num_expanded}")
        # print(f"Original length {len(token_ids)-num_expanded}, new length {len(token_ids)}")
        # print(f"Attempted to expand by {expand_prop*100}%, expanded by {num_expanded/len(token_ids)*100:.2f}%")
        return token_ids

    def set_expansions(self):
        """
        Loads expansions dict from file if it exists, otherwise builds and saves it to file.
        """
        filename = "data/tokenizer_expansions.json"
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        json_file_path = os.path.join(current_directory, filename)

        # Check if the file exists
        if os.path.exists(json_file_path):
            print(f"Found '{filename}' at: {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                expansions = json.load(f)
                assert isinstance(expansions, dict), f"{filename} must be a dictionary."
                print(f"Successfully loaded {filename}.")
        else:
            expansions = self.build_expansions()
            # save the expansions to the file
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(expansions, f)
            print(f"Successfully saved {filename}.")
        self.expansions = expansions
        print(f"Successfully set self.expansions.")

    def build_expansions(self):
        """
        1. Builds `merges` (dict):
            Keys: tuples of token_ids.
            Values: token_id of the result of merging the two tokens.
            eg. merges = {(token_id1, token_id2): token_id}
        2. Builds `expansions` (dict):
            Keys: token_id.
            Values: list of tuples of token_ids that the key token_id can be split into.
            eg. expansions = {token_id: [(token_id1_1, token_id1_2), (token_id2_1, token_id2_2), ...]}
        """
        # 1. Build merges dict
        ttokenizer_byt2int = self.tokenizer._mergeable_ranks # Mapping of tokenizer's vocab {token_string: token_id} stored as {bytes: int}
        ttokenizer_tokens_as_tuples = [tuple(token) for token in list(ttokenizer_byt2int.keys())] # Needed to be able to use `in` operator
        merges = {}
        for i, (token_as_bytes, token_id) in tqdm(
            enumerate(ttokenizer_byt2int.items()),
            total=len(ttokenizer_byt2int),
            desc="Building tokenizer's merges",
            ):
            if i < 256:
                assert len(token_as_bytes) == 1
            else:
                # print(f"{i=}: {token_as_bytes.decode('utf-8')}")
                num_merges = 0
                ## split the token at each possible point and find a split/"merge" where both parts are already present earlier in the vocab.
                for j in range(1, len(token_as_bytes)):
                    first_part = token_as_bytes[:j]
                    second_part = token_as_bytes[j:]
                    # print(f"{j=}: {first_part.decode('utf-8')} + {second_part.decode('utf-8')}")
                    if tuple(first_part) in ttokenizer_tokens_as_tuples and tuple(second_part) in ttokenizer_tokens_as_tuples:
                        first_part_id = ttokenizer_byt2int[first_part]
                        second_part_id = ttokenizer_byt2int[second_part]
                        merges[(first_part_id, second_part_id)] = token_id
                        num_merges += 1
                        # print(f"{first_part}: {first_part_id} + {second_part}: {second_part_id} -> {token_as_bytes}: {token_id}")
                assert num_merges >= 1, f"No merge found for {token_as_bytes=}: {list(token_as_bytes)=}"
        
        # 2. Build expansions dict
        expansions = {}
        for k, v in merges.items():
            if v in expansions:
                expansions[v].append(k)
            else:
                expansions[v] = [k]
        return expansions