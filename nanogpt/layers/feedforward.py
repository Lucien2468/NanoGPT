import cupy as cp  # was: import numpy as np
from reversegradGPU import Tensor
# Removed white_box_ml import — Dense and ReLU inlined below so they use cp instead of np

class FeedForward:
    def __init__(self, embed_dim, expand=4,swiglu=True):
        self.swiglu = swiglu
        self.embed_dim = embed_dim
        if swiglu:
            self.W_gate = Tensor(cp.random.randn(embed_dim, embed_dim * expand) * 0.1)
            self.W_value = Tensor(cp.random.randn(embed_dim, embed_dim * expand) * 0.1)
            self.W_out = Tensor(cp.random.randn(embed_dim * expand, embed_dim) * 1)
        else:
            self.W1 = Tensor(cp.random.randn(embed_dim, embed_dim * expand) * 0.1)
            self.b1 = Tensor(cp.zeros((1, embed_dim * expand)))                   
            self.W2 = Tensor(cp.random.randn(embed_dim * expand, embed_dim) * 0.1)
            self.b2 = Tensor(cp.zeros((1, embed_dim)))                            
    def forward(self, x):
        if self.swiglu:
            gate = self._swish(x @ self.W_gate)
            value = x @ self.W_value
            out = (gate * value) @ self.W_out
            return out
        else:
            h = x @ self.W1 + self.b1
            relu = h.clip(0, cp.inf)
            return relu @ self.W2 + self.b2
    def _swish(self, x):
        return x * x.sigmoid()
    def get_weights(self):
        if self.swiglu:
            return [self.W_gate, self.W_value, self.W_out]
        else:
            return [self.W1, self.b1, self.W2, self.b2]