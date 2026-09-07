import numpy as np
from nanogpt.layers.normalization import LayerNorm, RMSNorm
from nanogpt.layers.attention import MultiHeadAttention
from nanogpt.layers.feedforward import FeedForward
from nanogpt.layers.embedding import Embedding
from nanogpt.layers.projection import OutputProjection
import time
import cupy as cp
def sync():
    cp.cuda.Stream.null.synchronize()
class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_expand, rope = True, pre_ln = False, rms = True, swiglu = False):
        self.pre_ln = pre_ln
        self.attention = MultiHeadAttention(embed_dim, num_heads, rope)
        normalization = LayerNorm if not rms else RMSNorm
        self.normalization1 = normalization(embed_dim)
        self.feedforward = FeedForward(embed_dim, ff_expand, swiglu=swiglu)
        self.normalization2 = normalization(embed_dim)
    def forward(self, x):
        if self.pre_ln:
            attn_output = self.attention.forward(self.normalization1.forward(x))
            x = x + attn_output
            ff_output = self.feedforward.forward(self.normalization2.forward(x))
            x = x + ff_output
        else:
            attn_output = self.attention.forward(x)
            x = self.normalization1.forward(x + attn_output)
            ff_output = self.feedforward.forward(x)
            x = self.normalization2.forward(x + ff_output)
        return x

    # Temporarily replace your TransformerBlock.forward with this instrumented version,
    # OR make a standalone function that mirrors it:

    def forward_profiled(self, x):
        sync(); t = time.perf_counter()
        attn_output = self.attention.forward(x)
        sync(); print(f"  attention:   {time.perf_counter()-t:.4f}s")

        sync(); t = time.perf_counter()
        tmp = x + attn_output
        sync(); print(f"  residual1:   {time.perf_counter()-t:.4f}s")

        sync(); t = time.perf_counter()
        x = self.layernorm1.forward(tmp)
        sync(); print(f"  layernorm1:  {time.perf_counter()-t:.4f}s")

        sync(); t = time.perf_counter()
        ff_output = self.feedforward.forward(x)
        sync(); print(f"  feedforward: {time.perf_counter()-t:.4f}s")

        sync(); t = time.perf_counter()
        tmp = x + ff_output
        sync(); print(f"  residual2:   {time.perf_counter()-t:.4f}s")

        sync(); t = time.perf_counter()
        x = self.layernorm2.forward(tmp)
        sync(); print(f"  layernorm2:  {time.perf_counter()-t:.4f}s")
        return x
class Transformer:
    def __init__(self, vocab_size, embed_dim, num_heads, ff_expand, num_layers, rope = True, pre_ln = False, rms = True, swiglu = False):
        self.rms = rms
        self.pre_ln = pre_ln
        self.embedding = Embedding(vocab_size, embed_dim)
        self.projection = OutputProjection(embed_dim, vocab_size)
        self.layers = [TransformerBlock(embed_dim, num_heads, ff_expand, rope, pre_ln, rms, swiglu) for _ in range(num_layers)]
        if pre_ln: self.projection_LN = LayerNorm(embed_dim) if not rms else RMSNorm(embed_dim)
        self.weights=self._get_weights()
    def forward(self, x):
        x = self.embedding.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.projection.forward(self.projection_LN.forward(x) if self.pre_ln else x)
        return x
    def _get_weights(self):
        weights=[]
        weights.extend([self.embedding.weights, self.projection.weights])
        for block in self.layers:
            attention = block.attention
            weights.append(attention.W_O)
            for head in attention.attn_heads: weights.extend([head.weight_Q,head.weight_V,head.weight_K])
            feedforward = block.feedforward
            weights.extend(feedforward.get_weights())
            weights.extend(block.normalization1.get_weights()+block.normalization2.get_weights())
        if self.pre_ln: weights.extend(self.projection_LN.get_weights())
        return weights
