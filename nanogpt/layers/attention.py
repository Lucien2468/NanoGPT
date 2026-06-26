import numpy as np
from reversegrad import Tensor
from .positional_encoding import AliBiPositionalEncoding
class Attention:
    def __init__(self, embed_dim, num_heads=8, head_idx=0):
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.head_dim = int(embed_dim/num_heads)
        self.positional_encoding = AliBiPositionalEncoding(max_seq_len=512, num_heads=num_heads)
        self.weight_Q = Tensor(np.random.randn(embed_dim, self.head_dim) * 0.01)
        self.weight_K = Tensor(np.random.randn(embed_dim, self.head_dim) * 0.01)
        self.weight_V = Tensor(np.random.randn(embed_dim, self.head_dim) * 0.01)
    def forward(self, x):
        Q, K, V =  x @ self.weight_Q, x @ self.weight_K, x @ self.weight_V
        alibi_bias = self.positional_encoding.get_positional_encoding(Q.data.shape[0], head_idx=self.head_idx)
        scores = (Q @ K.reshape((K.data.shape[1], K.data.shape[0])) - alibi_bias) / Tensor(np.atleast_1d(np.sqrt(self.head_dim)))
        INF = 1e+10
        scores = scores + Tensor(np.triu(np.full(scores.data.shape, -INF)))
        scores = scores.softmax()
        with open("D:/NanoGPT/logs/scores.txt", "a") as file:
            file.write(f"{scores.data}\n")
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
    def forward(self, x):
        attn_outputs = []
        for i in range(self.num_heads):
            attn = self.attn_heads[i]
            result = attn.forward(x)
            attn_outputs.append(result)

        out = attn_outputs[0].concat(*attn_outputs[1:], axis=-1)
        return out
    