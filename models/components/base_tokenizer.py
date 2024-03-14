"""
A simple wrapper around the GPT2 Tokenizer to
standardize the interface for tokenization.
"""

import tiktoken
import torch

class BaseTokenizer:
    """A simple wrapper around the GPT2 BPE Tokenizer."""

    def __init__(self):
        gpt2_tokenizer = tiktoken.get_encoding("gpt2")
        self.tokenizer = gpt2_tokenizer
        self.eot_token = gpt2_tokenizer.eot_token
        self.pad_token = gpt2_tokenizer.eot_token
        self.vocab_size = gpt2_tokenizer.max_token_value + 1

    def pad_batch(self, token_lists, direction="right"):
        """Pad a list of token lists to the same length,
        and return the padded tensor, and mask tensor.
        
        Direction can be 'right' or 'left' to specify the padding direction.
        """
        max_len = max(len(tokens) for tokens in token_lists)
        padded_tokens = []
        mask = []
        for tokens in token_lists:
            if direction == "right":
                padded_tokens.append(tokens + [self.pad_token] * (max_len - len(tokens)))
                mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
            elif direction == "left":
                padded_tokens.append([self.pad_token] * (max_len - len(tokens)) + tokens)
                mask.append([0] * (max_len - len(tokens)) + [1] * len(tokens))
        return torch.tensor(padded_tokens), torch.tensor(mask)

    def encode(self, text):
        """Encode a string into tokens."""
        return self.tokenizer.encode_ordinary(text)

    def encode_batch(self, texts):
        """Encode a list of strings into tokens."""
        return self.tokenizer.encode_ordinary_batch(texts)

    def decode(self, tokens):
        """Decode a list of tokens into a string."""
        # check if the tokens are a tensor
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def decode_batch(self, token_lists):
        """Decode a list of token lists into a list of strings."""
        if torch.is_tensor(token_lists):
            token_lists = token_lists.tolist()
        return self.tokenizer.decode_batch(token_lists)
