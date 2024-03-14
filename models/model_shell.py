"""
The standard Model Shell. It combines the embedding model,
core model and LM head.
"""

import torch

from models import core_models, embedding_models, model_heads



class ModelShell(torch.nn.Module):
    """
    Unify the embedding model, core model and LM head
    into a single object; initializes the weights
    and prints basic model statistics.
    """

    def __init__(
        self,
        embedding_model: embedding_models.EmbedderInterface,
        core_model: core_models.GenericTransformer,
        model_head: model_heads.AutoregressiveLMHead,
        weight_init_func=None,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.core_model = core_model
        self.model_head = model_head

        # initialize model weights
        if weight_init_func is not None:
            self.apply(weight_init_func)
        self.device = ...

    # override to device to set the attribute
    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def forward(self, token_ids, attention_mask=None):
        """
        The default forward pass is used for trianing and
        accepts the token_ids as input.
        """

        # pass the token_ids through the embedding model
        # to get B, S, H (with pos encoding if necessary)
        x = self.embedding_model(token_ids)

        # pass the embeddings through the core model
        x = self.core_model(x, attention_mask=attention_mask)

        # pass the core model output through the model head
        x = self.model_head(x)

        return x


    @torch.no_grad()
    def loglikelihood(self, prefixes, continuations):
        """
        Compute the loglikelihood of continuation
        tokens given a prefix.
        Args:
            prefixes: list[str]
            continuations: list[str]
        Returns:
            ll: torch.tensor(B)
        """
        total_strings = [f"{prefix} {cont}" for prefix, cont in zip(prefixes, continuations)]
        input_tokens = [self.embedding_model.tokenize_input(string, truncate=True) for string in total_strings]
        padded_batch, mask = self.embedding_model.pad_batch(input_tokens, direction="right")
        input_tensor = padded_batch.detach().clone().to(device=self.device, dtype=torch.long)
        logits, _ = self.forward(input_tensor)
        logits = logits[:, :-1].reshape(-1, logits.size(-1))
        target_tensor = input_tensor[:, 1:].reshape(-1)
        ll = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="none")
        mask = mask[:, 1:].reshape(-1).to(ll.device)
        ll = ll * mask
        ll = ll.view(input_tensor.size(0), -1).sum(dim=1)
        return -ll