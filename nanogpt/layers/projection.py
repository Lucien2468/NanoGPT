from reversegrad import Tensor
import cupy as cp  # was: import numpy as np
class OutputProjection:
    def __init__(self, embed_dim, vocab_size):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.weights = Tensor(cp.random.randn(embed_dim, vocab_size) * 0.01)  # np.random.randn → cp.random.randn
        self.grad = cp.zeros_like(self.weights.data)  # np.zeros_like → cp.zeros_like
    def forward(self, x):
        return (x @ self.weights)