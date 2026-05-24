import numpy as np
from reversegrad import Tensor
class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.weights = Tensor(np.random.randn(vocab_size, embed_dim) * 0.01)
        self._backward = lambda: None
        self._children = []
    def forward(self, tokens):
        if hasattr(tokens, 'data'):
            tokens = tokens.data
        return self.weights[np.asarray(tokens).astype(int)]