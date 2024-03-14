"""
Contains the build functions for the embedder,
core model, lm head and the model shell.
"""

import torch
from models.core_models import GenericTransformer
from models.embedding_models import Embedder
from models.model_heads import AutoregressiveLMHead
from models.model_shell import ModelShell
from models.components.base_tokenizer import BaseTokenizer


def build_model(model_cfg=None, checkpoint_path=None):
    """
    Either initialize or load a model, depending on
    whether a config or checkpoint was provided
    (respectively).
    Args:
        model_cfg: model_configuration
        model_checkpoint: model_checkpoint_dict
        dataset_name: the dataset for the tokenizer
    Returns:
        model: model instance
    """
    # check if model is to be loaded
    if checkpoint_path is not None:

        # load model with the correct architecture
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = initialize_model(checkpoint["config"]["model"])

        # load the model weights
        model.load_state_dict(checkpoint["model"])
        
    else:
        # initialize model
        print(f"Initializing model from scratch")
        model = initialize_model(model_cfg)

    return model

def initialize_model(model_cfg):
    """
    Initialize the model given the configuration.
    Args:
        model_cfg: model_cfg
    Returns:
        model: model_instance
    """
    # build the tokenizer
    tokenizer = BaseTokenizer()

    # build the embedding model
    embedding_model = Embedder(model_cfg=model_cfg, tokenizer=tokenizer)

    # build the core model
    core_model = GenericTransformer(model_cfg=model_cfg)

    # build the model head
    model_head = AutoregressiveLMHead(model_cfg=model_cfg)

    # check if embedding model weights are to be shared with the model head
    if model_cfg["embedding_weight_tying"]:
        # share the weights between the token embeddings and the final
        # logit layer, following: https://paperswithcode.com/method/weight-tying
        embedding_model.token_embedder.weight = model_head.linear.weight

    # build the model shell
    model = ModelShell(
        embedding_model=embedding_model, core_model=core_model, model_head=model_head
    )

    return model
