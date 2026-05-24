import numpy as np
from reversegrad import Tensor
from .positional_encoding import AliBiPositionalEncoding
class Attention:
    def __init__(self, embed_dim, num_heads=8, head_idx=0):
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.positional_encoding = AliBiPositionalEncoding(max_seq_len=512, num_heads=num_heads)
        self.weight_Q = Tensor(np.random.randn(embed_dim, embed_dim) * 0.01)
        self.weight_K = Tensor(np.random.randn(embed_dim, embed_dim) * 0.01)
        self.weight_V = Tensor(np.random.randn(embed_dim, embed_dim) * 0.01)
    def forward(self, x):
        Q, K, V =  x @ self.weight_Q, x @ self.weight_K, x @ self.weight_V
        alibi_bias = self.positional_encoding.get_positional_encoding(Q.data.shape[0], head_idx=self.head_idx)
        scores = (Q @ K.reshape((K.data.shape[1], K.data.shape[0])) + alibi_bias) / Tensor(np.atleast_1d(np.sqrt(self.embed_dim)))
        scores = scores.softmax()
        out = scores @ V
        return out

class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_heads = []
    def forward(self, x):
        batch_size = x.data.shape[0]
        x = x.reshape(batch_size, self.num_heads, self.head_dim)
        attn_outputs = []
        for i in range(self.num_heads):
            self.attn_heads.append(Attention(self.head_dim, num_heads=self.num_heads, head_idx=i))
            attn = self.attn_heads[i]
            result = attn.forward(x[:, i, :])
            attn_outputs.append(result)

        out = attn_outputs[0].concat(*attn_outputs[1:], axis=-1)
        return out
    