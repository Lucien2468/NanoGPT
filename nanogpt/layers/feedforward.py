import cupy as cp  # was: import numpy as np
from reversegrad import Tensor
# Removed white_box_ml import — Dense and ReLU inlined below so they use cp instead of np

class FeedForward:
    def __init__(self, embed_dim, expand=4):
        self.embed_dim = embed_dim
        # Two linear layers: embed_dim → embed_dim*expand → embed_dim
        self.W1 = Tensor(cp.random.randn(embed_dim, embed_dim * expand) * 0.1)  # was Dense(embed_dim, embed_dim*expand).weights
        self.b1 = Tensor(cp.zeros((1, embed_dim * expand)))                      # was Dense.biases
        self.W2 = Tensor(cp.random.randn(embed_dim * expand, embed_dim) * 0.1)  # was Dense(embed_dim*expand, embed_dim).weights
        self.b2 = Tensor(cp.zeros((1, embed_dim)))                               # was Dense.biases

    def forward(self, x):
        h = x @ self.W1 + self.b1          # first linear layer (same as Dense.forward)
        # ReLU: cp.maximum replaces np.maximum; backward wired manually like activations.py did
        relu = Tensor(cp.maximum(0, h.data))  # np.maximum → cp.maximum
        def _relu_backward():
            h.grad += relu.grad * (h.data > 0)  # derivative of max(0,x) is 1 where x>0
        relu._backward = _relu_backward
        relu._children = [h]
        return relu @ self.W2 + self.b2    # second linear layer
