import numpy as np
from nanogpt.layers.layernorm import LayerNorm
from nanogpt.layers.attention import MultiHeadAttention
from nanogpt.layers.feedforward import FeedForward
from nanogpt.layers.embedding import Embedding
from nanogpt.layers.projection import OutputProjection
class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.layernorm1 = LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim)
        self.layernorm2 = LayerNorm(embed_dim)
    def forward(self, x):
        attn_output = self.attention.forward(x)
        x = self.layernorm1.forward(x + attn_output)
        ff_output = self.feedforward.forward(x)
        x = self.layernorm2.forward(x + ff_output)
        return x
class Transformer:
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.projection = OutputProjection(embed_dim, vocab_size)
        self.layers = [TransformerBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)]
    def forward(self, x):
        x = self.embedding.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.projection.forward(x)
        return x