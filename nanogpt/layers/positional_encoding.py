import numpy as np
from reversegrad import Tensor
class AliBiPositionalEncoding:
    def __init__(self, max_seq_len, num_heads):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.alibi_slopes = self._get_alibi_slopes(num_heads)
    def get_positional_encoding(self, seq_len, head_idx):
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        alibi = np.zeros((seq_len,seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if not j > i:
                    alibi[i,j] = (i-j) * self.alibi_slopes[head_idx]
        return Tensor(alibi)

    def _get_alibi_slopes(self, num_heads):
        slopes = []
        for i in range(num_heads):
            slope = 1.0 / (2 ** (i+1))
            slopes.append(slope)
        return slopes