from nanogpt import MultiHeadAttention, Embedding, FeedForward, LayerNorm, Transformer
import numpy as np
# Test the implementation
if __name__ == "__main__":
    vocab_size = 100
    embed_dim = 64
    num_heads = 8
    ff_hidden_dim = 256
    num_layers = 2
    transformer = Transformer(vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers)
    input_data = np.random.randint(0, vocab_size, (10, 20))
    output = transformer.forward(input_data)
    print("Output shape:", output.data.shape)