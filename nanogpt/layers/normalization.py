
import cupy as cp  # was: import numpy as np
from reversegradGPU import Tensor
class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.embed_dim = embed_dim
        self.eps = cp.asarray(eps)           # np.asarray → cp.asarray
        self.gamma = Tensor(cp.ones((1, embed_dim)))   # np.ones → cp.ones
        self.beta = Tensor(cp.zeros((1, embed_dim)))   # np.zeros → cp.zeros
    def forward(self, x):
        x = x if hasattr(x, 'data') else Tensor(x)
        self.x = x
        mean = x.mean(axis=-1, keepdims=True)

        var = x.var(axis=-1, keepdims=True)
        normalized_numerator = x - mean
        variance_with_eps = var + self.eps
        std = variance_with_eps ** Tensor(0.5)
        self.x_normalized = normalized_numerator / std
        return self.gamma * self.x_normalized + self.beta
    def get_weights(self):
        return [self.gamma, self.beta]
'''
    def backward(self, grad_output):
        n = self.x_normalized.data.shape[-1]
        std = np.sqrt(np.var(self.x.data, axis=-1, keepdims=True) + self.eps.data)

        self.gamma_grad = np.sum(grad_output * self.x_normalized.data, axis=0)
        self.beta_grad = np.sum(grad_output, axis=0)
        self.x_grad = (1.0 / (n * std)) * (n * grad_output * self.gamma.data - np.sum(grad_output * self.gamma.data, axis=-1, keepdims=True) - self.x_normalized.data * np.sum(grad_output * self.gamma.data * self.x_normalized.data, axis=-1, keepdims=True))

import numpy as np
class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = np.ones((embed_dim,))
        self.beta = np.zeros((embed_dim,))
    def forward(self, x):
        self.x = x
        mean = x.mean(axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.x_normalized = (x - mean) / (var + np.full(var.shape, self.eps)) ** np.full(var.shape, 1/2)
        return self.gamma * self.x_normalized + self.beta

    def backward(self, grad_output):
        mean = self.x_normalized.mean(axis=-1, keepdims=True)
        var = np.var(self.x, axis=-1, keepdims=True)
        n = self.x_normalized.shape[-1]
        std = (np.sqrt(np.var(self.x, axis=-1, keepdims=True) + self.eps))

        self.gamma_grad = np.sum(grad_output * self.x_normalized, axis=0)
        self.beta_grad = np.sum(grad_output, axis=0)
        self.x_grad = (1.0 / (n * std)) * (n * grad_output * self.gamma - np.sum(grad_output * self.gamma, axis=-1, keepdims=True) - self.x_normalized * np.sum(grad_output * self.gamma * self.x_normalized, axis=-1, keepdims=True))
'''
class RMSNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.embed_dim = embed_dim
        self.eps = cp.asarray(eps)           # np.asarray → cp.asarray
        self.gamma = Tensor(cp.ones((1, embed_dim)))   # np.ones → cp.ones
    def forward(self, x):
        x = x if hasattr(x, 'data') else Tensor(x)

        self.x_normalized = x / self._rms(x) + self.eps
        return self.gamma * self.x_normalized
    def _rms(self, x):
        x_squared = x ** Tensor(2)
        x_squared_mean = x_squared.mean(axis=-1, keepdims = True)
        return x_squared_mean ** Tensor(0.5)
    def get_weights(self):
        return [self.gamma]