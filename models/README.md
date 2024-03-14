# Super Tiny Language Models / Models

A set of basic components for tokenizers and transformer layers.

## Interfaces
We provide the basic interface that all models must meet in [model_shell.py](model_shell.py).
The shell is assumed to contain three components:
1. **Embedder**: This component takes care of both tokenizing input, and also embedding the tokens into a dense, continuous representation that can be processed by the transformer layers. The embedder interface is given in [embedding_models.py](embedding_models.py).
2. **Transformer Core**: This component is the core of the model, and typically consists in a stack of transformer layers. We don't assume any particular interface for this, however we do implement a [`generic transformer'](core_models.py) that is intended to subsume most use cases.
3. **LM Head**: This component takes the output of the transformer core and maps it to the output space of the model. We define the interface in [model_heads.py](model_heads.py).

## Other Components
2. **Normalization**:
In [normalization.py](components/layers/normalization.py) we implement RMSNorm, LayerNorm, and a pass-through layer.
3. **Positional Encodings**:
In [positional_encodings.py](components/positional_encodings.py) we implement a variety of positional encodings, including the standard sinusoidal positional encodings, and the relative positional encodings from Shaw et al. (2018).
5. **Attention**:
Our [attention layer](components/layers/attention.py) implements, causal masks, groups, multi-head, and rotary embeddings.
6. **Feed Forward**:
In [feed_forward.py](components/layers/feedforward.py) we implement both the standard feedforward layer (with variable [activation](components/layers/activations.py) as well as the SwiGLU activation from [Shazeer et al. (2020)](
https://arxiv.org/abs/2002.05202).)
