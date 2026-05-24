import os
import numpy as np
class Indicer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        path_to_indices = "D:/NanoGPT/data/indices.npy"
        if os.path.exists(path_to_indices):
            self.indices = np.load(path_to_indices, allow_pickle=True).item()
        else:
            self.indices = {}
            np.save(path_to_indices, self.indices)
    def add_data(self, data):
        for token in data:
            if token not in self.indices:
                self.indices[token] = len(self.indices)
        np.save("D:/NanoGPT/data/indices.npy", self.indices)
    def encode(self, tokens):
        return np.array([self.indices.get(token, 0) for token in tokens])
    def decode(self, indices):
        reverse = {}
        for token, index in self.indices.items():
            reverse[index] = token
        result = []
        for i in indices:
            if i in reverse:
                result.append(reverse[i])
            else:
                result.append("<unk>")
        return result
data = np.load("D:/NanoGPT/data/indexed_tokens.npy", allow_pickle=True)
tokens = [item['token'] for item in data]
vocab_size = len(set(tokens))
indicer = Indicer(vocab_size)
indicer.add_data(tokens)
encoded_tokens = indicer.encode(tokens)
print("Encoded Tokens:", encoded_tokens[:10])
np.save("D:/NanoGPT/data/encoded_tokens.npy", encoded_tokens)
decoded_tokens = indicer.decode(encoded_tokens)
print("Decoded Tokens:", decoded_tokens[:10])
print("Vocabulary Size:", vocab_size)