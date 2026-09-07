import cupy as cp  # was: import numpy as np
from reversegradGPU import Tensor
class AliBiPositionalEncoding:
    def __init__(self, num_heads):
        self.num_heads = num_heads
        self.alibi_slopes = self._get_alibi_slopes(num_heads)
    def get_positional_encoding(self, seq_len, head_idx):
        positions = cp.arange(seq_len, dtype='float32')
        diff = positions[:, None] - positions[None, :]        # (seq, seq), [i,j] = i - j
        alibi = diff * self.alibi_slopes[head_idx]
        alibi = cp.tril(alibi)                                # keep only j <= i
        return Tensor(alibi)
    def _get_alibi_slopes(self, num_heads):
        slopes = []
        for i in range(num_heads):
            slope = 1.0 / (2 ** (i+1))
            slopes.append(slope)
        return cp.asarray(slopes, dtype='float32')
    
class RoPEPositionalEncoding:
    def __init__(self, head_dim):
        self.head_dim = head_dim
    def get_positional_encoding(self, seq_len):
        theta = 10000 ** (-2 * cp.arange(self.head_dim // 2, dtype='float32') / self.head_dim)
        positions = cp.arange(seq_len, dtype='float32')
        angles = positions[:, None] * theta
        return angles