import numpy as np
from nanogpt.layers.layernorm import LayerNorm
from nanogpt.layers.attention import MultiHeadAttention
from nanogpt.layers.feedforward import FeedForward
from nanogpt.layers.embedding import Embedding
from nanogpt.layers.projection import OutputProjection
class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_expand):

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.layernorm1 = LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, ff_expand)
        self.layernorm2 = LayerNorm(embed_dim)
    def forward(self, x):
        attn_output = self.attention.forward(x)
        x = self.layernorm1.forward(x + attn_output)
        ff_output = self.feedforward.forward(x)
        x = self.layernorm2.forward(x + ff_output)
        return x
class Transformer:
    def __init__(self, vocab_size, embed_dim, num_heads, ff_expand, num_layers):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.projection = OutputProjection(embed_dim, vocab_size)
        self.layers = [TransformerBlock(embed_dim, num_heads, ff_expand) for _ in range(num_layers)]
        self.weights=self._get_weights()
    def forward(self, x):
        x = self.embedding.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.projection.forward(x)
        return x
    def _get_weights(self):
        weights=[]
        weights.extend([self.embedding.weights, self.projection.weights])
        for block in self.layers:
            attention = block.attention
            weights.append(attention.W_O)
            for head in attention.attn_heads: weights.extend([head.weight_Q,head.weight_V,head.weight_K])
            feedforward = block.feedforward
            weights.extend([feedforward.W1, feedforward.b1, feedforward.W2, feedforward.b2])  # was feedforward.model.layers loop — FeedForward now has W1/b1/W2/b2 directly
            weights.extend([block.layernorm1.gamma,block.layernorm1.beta,block.layernorm2.gamma,block.layernorm2.beta])
        return weights