import cupy as cp  # was: import numpy as np
from reversegradGPU import Tensor
from .positional_encoding import AliBiPositionalEncoding, RoPEPositionalEncoding
class Attention:
    def __init__(self, embed_dim, num_heads=8, head_idx=0, rope=True):
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.head_dim = int(embed_dim/num_heads)
        self.positional_encoding = RoPEPositionalEncoding(head_dim=self.head_dim) if rope else AliBiPositionalEncoding(num_heads=num_heads)
        self.rope = rope
        self.weight_Q = Tensor(cp.random.randn(embed_dim, self.head_dim) * 0.01)  # np.random.randn → cp.random.randn
        self.weight_K = Tensor(cp.random.randn(embed_dim, self.head_dim) * 0.01)  # np.random.randn → cp.random.randn2ws
        self.weight_V = Tensor(cp.random.randn(embed_dim, self.head_dim) * 0.01)  # np.random.randn → cp.random.randn
    def forward(self, x):
        Q, K, V =  x @ self.weight_Q, x @ self.weight_K, x @ self.weight_V
        if self.rope:
            Q_rot, K_rot = self._rotate(Q, K)
            encoded = Q_rot @ K_rot.swapaxes(-1,-2) 
        else:
            encoded = Q @ K.swapaxes(-1,-2) - self.positional_encoding.get_positional_encoding(Q.data.shape[-2], head_idx=self.head_idx)
        scores = encoded / Tensor(cp.atleast_1d(cp.sqrt(cp.float32(self.head_dim))))  # np.atleast_1d/np.sqrt → cp.atleast_1d/cp.sqrt
        INF = 1e+10
        scores = scores + Tensor(cp.triu(cp.full(scores.data.shape, -INF), k=1))  # np.triu/np.full → cp.triu/cp.full
        scores = scores.softmax()
        self.scores = scores
        out = scores @ V
        return out
    def _rotate(self, Q, K):
        Q_1, Q_2 = Q[..., :self.head_dim//2], Q[..., self.head_dim//2:]
        K_1, K_2 = K[..., :self.head_dim//2], K[..., self.head_dim//2:]
        angles = self.positional_encoding.get_positional_encoding(Q.data.shape[-2])
        angles_sin = cp.sin(angles)
        angles_cos = cp.cos(angles)
        Q_1_rot, Q_2_rot = Q_1 * angles_cos - Q_2 * angles_sin, Q_1 * angles_sin + Q_2 * angles_cos
        K_1_rot, K_2_rot = K_1 * angles_cos - K_2 * angles_sin, K_1 * angles_sin + K_2 * angles_cos
        Q_rot = Q_1_rot.concat(Q_2_rot, axis=-1)
        K_rot = K_1_rot.concat(K_2_rot, axis=-1)
        return Q_rot, K_rot
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads, rope=True):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_heads = []
        for i in range(self.num_heads):
            self.attn_heads.append(Attention(self.embed_dim, num_heads=self.num_heads, head_idx=i, rope=rope))
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