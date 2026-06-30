import cupy as cp  # was: import numpy as np
from reversegrad import Tensor
from .positional_encoding import AliBiPositionalEncoding
class Attention:
    def __init__(self, embed_dim, num_heads=8, head_idx=0):
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.head_dim = int(embed_dim/num_heads)
        self.positional_encoding = AliBiPositionalEncoding(max_seq_len=512, num_heads=num_heads)
        self.weight_Q = Tensor(cp.random.randn(embed_dim, self.head_dim) * 0.01)  # np.random.randn → cp.random.randn
        self.weight_K = Tensor(cp.random.randn(embed_dim, self.head_dim) * 0.01)  # np.random.randn → cp.random.randn
        self.weight_V = Tensor(cp.random.randn(embed_dim, self.head_dim) * 0.01)  # np.random.randn → cp.random.randn
    def forward(self, x):
        Q, K, V =  x @ self.weight_Q, x @ self.weight_K, x @ self.weight_V
        alibi_bias = self.positional_encoding.get_positional_encoding(Q.data.shape[-2], head_idx=self.head_idx)
        scores = (Q @ K.swapaxes(-1,-2) - alibi_bias) / Tensor(cp.atleast_1d(cp.sqrt(cp.float64(self.head_dim))))  # np.atleast_1d/np.sqrt → cp.atleast_1d/cp.sqrt
        INF = 1e+10
        scores = scores + Tensor(cp.triu(cp.full(scores.data.shape, -INF), k=1))  # np.triu/np.full → cp.triu/cp.full
        scores = scores.softmax()
        out = scores @ V
        return out

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_heads = []
        for i in range(self.num_heads):
            self.attn_heads.append(Attention(self.embed_dim, num_heads=self.num_heads, head_idx=i))
        self.W_O = Tensor(cp.random.randn(embed_dim, embed_dim) * 0.01)  # np.random.randn → cp.random.randn
    def forward(self, x):
        attn_outputs = []
        for i in range(self.num_heads):
            attn = self.attn_heads[i]
            result = attn.forward(x)
            attn_outputs.append(result)

        out = attn_outputs[0].concat(*attn_outputs[1:], axis=-1)
        out = out @ self.W_O
        return out
    