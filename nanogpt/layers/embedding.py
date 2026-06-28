import cupy as cp  # was: import numpy as np
from reversegrad import Tensor
class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.weights = Tensor(cp.random.randn(vocab_size, embed_dim) * 0.01)  # np.random.randn → cp.random.randn
        self._backward = lambda: None
        self._children = []
    def forward(self, tokens):
        if hasattr(tokens, 'data'):
            tokens = tokens.data if isinstance(tokens,Tensor) else tokens
        return self.weights[cp.asarray(tokens).astype(int)]  # np.asarray → cp.asarray: token indices on GPU