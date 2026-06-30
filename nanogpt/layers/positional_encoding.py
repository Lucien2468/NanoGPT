import cupy as cp  # was: import numpy as np
from reversegrad import Tensor
class AliBiPositionalEncoding:
    def __init__(self, max_seq_len, num_heads):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.alibi_slopes = self._get_alibi_slopes(num_heads)

    def get_positional_encoding(self, seq_len, head_idx):
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        positions = cp.arange(seq_len)
        diff = positions[:, None] - positions[None, :]        # (seq, seq), [i,j] = i - j
        alibi = diff * self.alibi_slopes[head_idx]
        alibi = cp.tril(alibi)                                # keep only j <= i
        return Tensor(alibi)
    def _get_alibi_slopes(self, num_heads):
        slopes = []
        for i in range(num_heads):
            slope = 1.0 / (2 ** (i+1))
            slopes.append(slope)
        return slopes