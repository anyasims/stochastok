"""
Simple, flexible core models.
"""

import torch

from models.components.layers.transformer_blocks import GenericTransformerBlock


class GenericTransformer(torch.nn.Module):
    """
    Generic Transformer Class intended to be used for as
    broad a range of transformer models as possible.
    """

    def __init__(self, model_cfg):
        super().__init__()

        # build the transformer
        self.transformer = torch.nn.ModuleDict(
            {
                "drop": torch.nn.Dropout(),
                "h": torch.nn.ModuleList(
                    [
                        GenericTransformerBlock(
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"],
                            use_rope=model_cfg["positional_encoding_type"] == "rope",
                            ffn_cfg=model_cfg["core_model"]["ffn"],
                            attn_cfg=model_cfg["core_model"]["attn"],
                        )
                        for _ in range(model_cfg["core_model"]["num_layers"])
                    ]
                ),
            }
        )

    def forward(self, x, attention_mask=None):
        """
        Pass an input through the model
        Args:
            x: torch.tensor(B, S, H)
        Returns:
            x: torch.tensor(B, S, H)
        """

        # apply dropout
        x = self.transformer.drop(x)

        # pass through the transformer blocks
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)

        return x

