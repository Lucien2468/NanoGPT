import os
from collections import Counter
import numpy as np
class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    def split_map(self, text):
        tokens = text.split()
        unique_tokens = set(tokens)
        return unique_tokens
    def sequentialize(self, text):
        return text.split()
class Indicer:
    def __init__(self, vocab_size, path_to_indices="D:/NanoGPT/data/indices.npy"):
        self.vocab_size = vocab_size
        np.save(path_to_indices, {})  # Initialize with an empty dictionary
        self.map = np.load(path_to_indices, allow_pickle=True).item() if os.path.exists(path_to_indices) else {}
    def fit(self, tokens):
        unique_ordered = unique_ordered = [item for item, count in Counter(tokens).most_common()]
        for i, token in enumerate(list(unique_ordered)):
            if i < self.vocab_size:
                self.map[token] = i+1
        np.save("D:/NanoGPT/data/indices.npy", self.map)
    def encode(self, tokens):
        return [self.map.get(token, 0) for token in tokens]
    def decode(self, indices):
        return [list(self.map.keys())[list(self.map.values()).index(index)] if index in self.map.values() else "<unk>" for index in indices]
class Indexer:
    def __init__(self, text_size, path_to_indexes="D:/NanoGPT/data/indexes.npy", path_to_indices="D:/NanoGPT/data/indices.npy"):
        self.text_size = text_size
        self.path_to_indexes = path_to_indexes
        self.indexes = np.load(path_to_indexes, allow_pickle=True) if os.path.exists(path_to_indexes) else np.array([], dtype=[('token', 'U256'), ('index', 'i4'), ('indice', 'i4')])
        self.indices = np.load(path_to_indices, allow_pickle=True).item() if os.path.exists(path_to_indices) else {}
    def fit(self, tokens, text=None):
        vocab = {}
        for i, token in enumerate(tokens):
            vocab[token] = i

        if text is None:
            result = []
            for row in self.indexes:
                result.append(int(row['index']))
            return result

        text_tokens = text.split()
        rows = []
        for i, token in enumerate(text_tokens):
            if token in vocab:
                index = vocab[token]
            else:
                index = 0
            rows.append((token, index, i))

        records = np.array(rows, dtype=[('token', 'U256'), ('index', 'i4'), ('indice', 'i4')])
        np.save(self.path_to_indexes, records)
        self.indexes = records
    def encode(self, tokens):
        result = []
        for token in tokens:
            found = False
            for row in self.indexes:
                if row['token'] == token:
                    result.append(int(row['index']))
                    found = True
                    break
            if not found:
                result.append(0)
        return result
    def decode(self, indices):
        result = []
        for idx in indices:
            found = False
            for row in self.indexes:
                if int(row['index']) == idx:
                    result.append(row['token'])
                    found = True
                    break
            if not found:
                result.append('<unk>')
        return result

