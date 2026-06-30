import cupy as cp  # was: import numpy as np
import numpy as np
from reversegrad import Tensor
class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.weights = Tensor(cp.random.randn(vocab_size, embed_dim) * 0.01)  # np.random.randn → cp.random.randn
        self._backward = lambda: None
        self._children = []
    def forward(self, tokens):
        if isinstance(tokens, Tensor):
            tokens = tokens.data
        # convert whole array to clean int64, works for ANY shape (1D or 2D batched):
        if isinstance(tokens, cp.ndarray):
            tokens_int = tokens.astype(cp.int64)        # already on GPU, just cast
        else:
            tokens_int = cp.asarray(np.asarray(tokens, dtype=np.int64))  # CPU list/array → GPU int
        return self.weights[tokens_int]