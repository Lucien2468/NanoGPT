import os
from collections import Counter
import numpy as np
class Indicer:
    def __init__(self, vocab_size, gaps = [], map = {}):
        self.gaps = gaps
        self.vocab_size = vocab_size
        self.map = map
    def fit(self, tokens):
        unique_ordered = [item for item, count in Counter(tokens).most_common()]
        for i, token in enumerate(list(unique_ordered)):
            if i < self.vocab_size:
                self.map[token] = i+1
        for i, gap in enumerate(self.gaps):
            self.map[gap] = i+self.vocab_size+1
        return self.map
    def encode(self, tokens):
        return [self.map.get(token, 0) for token in tokens]
    def decode(self, indices):
        return [list(self.map.keys())[list(self.map.values()).index(index)] if index in self.map.values() else "<unk>" for index in indices]
    

class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merge_list = []
        self.map = {"<unk>":0}
    def train(self, corpus, num_merges, verbose = False):
        words = corpus.split()
        working_corpus = [[" "] + list(word) for word in words]
        all_chars = [char for word in working_corpus for char in word]
        unique_ordered = [item for item, count in Counter(all_chars).most_common()]
        for i, token in enumerate(unique_ordered):
            self.map[token] = i + 1
        for _ in range(num_merges):
            pair_counts = Counter()
            for word in working_corpus:
                for pair in zip(word[:-1], word[1:]):
                    pair_counts[pair] += 1
            pair_merge = pair_counts.most_common()[0][0]
            self.merge_list.append(pair_merge)
            merge = pair_merge[0] + pair_merge[1]
            self.map[merge] = len(self.map)+1
            new_corpus = []
            for word in working_corpus:
                new_word = []
                i = 0
                while i < len(word) - 1:
                    if word[i] == pair_merge[0] and word[i+1] == pair_merge[1]:
                        new_word.append(merge)
                        i+=2
                    else:
                        new_word.append(word[i])
                        i+=1
                if i+1 == len(word):
                    new_word.append(word[-1])
                new_corpus.append(new_word)
            working_corpus = new_corpus.copy()
            if verbose:
                print(_)
    def encode(self, tokens):
        chars = tokens.split()
        working_words = [[" "] + list(word) for word in chars]

        for merge in self.merge_list:
            merged_pair = merge[0] + merge[1]
            new_words = []
            for word in working_words:
                new_chars = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == merge[0] and word[i+1] == merge[1]:
                        new_chars.append(merged_pair)
                        i += 2
                    else:
                        new_chars.append(word[i])
                        i += 1
                new_words.append(new_chars)

            working_words = new_words

        flat_tokens = [token for word in working_words for token in word]

        return [self.map.get(token, 0) for token in flat_tokens]
    def decode(self, indices):
        rev_map = {v: k for k, v in self.map.items()}
        return "".join([rev_map.get(indice) for indice in indices])