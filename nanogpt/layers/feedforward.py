from reversegrad import Tensor
from white_box_ml.nn.activations import ReLU
from white_box_ml.nn.layer import Dense
from white_box_ml.nn.sequential import Sequential
import numpy as np
class FeedForward:
    def __init__(self, embed_dim, expand=4):
        self.embed_dim = embed_dim
        self.model = Sequential()
        self.model.add(Dense(self.embed_dim, self.embed_dim * expand), ReLU())
        self.model.add(Dense(self.embed_dim * expand, self.embed_dim), None)
    def forward(self, x):
        return self.model.forward(x)