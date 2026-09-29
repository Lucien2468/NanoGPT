from reversegradGPU import Tensor
import cupy as cp  # was: import numpy as np
class OutputProjection:
    def __init__(self, embed_dim, vocab_size, reward_model = False):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.weights = Tensor(cp.random.randn(embed_dim, vocab_size) * 0.01)  if not reward_model else Tensor(cp.random.randn(embed_dim, 1) * 0.01)
    def forward(self, x):
        return x @ self.weights