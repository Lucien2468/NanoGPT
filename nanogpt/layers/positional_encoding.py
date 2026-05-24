import numpy as np
from reversegrad import Tensor
class AliBiPositionalEncoding:
    def __init__(self, max_seq_len, num_heads):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.alibi_slopes = self._get_alibi_slopes(num_heads)

    def get_positional_encoding(self, seq_len, head_idx=None):
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        heads = [head_idx] if head_idx is not None else range(self.num_heads)
        n = len(heads) if head_idx is None else 1
        alibi = np.zeros((n, seq_len, seq_len))
        for out_idx, h in enumerate(heads):
            slope = self.alibi_slopes[h]
            for i in range(seq_len):
                for j in range(i + 1):
                    alibi[out_idx, i, j] = slope * (i - j)
        if head_idx is not None:
            return Tensor(alibi[0])  # shape (seq_len, seq_len)
        return Tensor(alibi)  # shape (num_heads, seq_len, seq_len)
    def _get_alibi_slopes(self, num_heads):
        slopes = []
        for i in range(num_heads):
            slope = 1.0 / (2 ** (i / num_heads))
            slopes.append(slope)
        return slopes