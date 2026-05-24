from reversegrad import Tensor
from nanogpt.layers.layernorm import LayerNorm
import numpy as np
class Optimizer:
    def __init__(self, loss_function,model , max_iter=100):
        self.loss_function = loss_function
        self.max_iter = max_iter
        self.model = model
        self.layers = self.model.layers
        self.embedding = self.model.embedding
        self.projection = self.model.projection
    def step(self, lr):
        self.embedding.weights -= Tensor(lr * self.embedding.weights.grad)
        self.projection.weights -= Tensor(lr * self.projection.weights.grad)
        for block in self.layers:
            attention = block.attention
            for head in attention.attn_heads:
                head.weight_Q -= Tensor(lr * head.weight_Q.grad)
                head.weight_V -= Tensor(lr * head.weight_V.grad)
                head.weight_K -= Tensor(lr * head.weight_K.grad)
            feedforward = block.feedforward
            for layer, _ in feedforward.model.layers:
                layer.weights -= Tensor(lr * layer.weights.grad)

                layer.biases  -= Tensor(lr * layer.biases.grad)
            block.layernorm1.gamma -= Tensor(lr * block.layernorm1.gamma.grad)
            block.layernorm1.beta -= Tensor(lr * block.layernorm1.beta.grad)
            block.layernorm2.gamma -= Tensor(lr * block.layernorm2.gamma.grad)
            block.layernorm2.beta -= Tensor(lr * block.layernorm2.beta.grad)
    def zero_grad(self):
        self.embedding.weights.grad = np.zeros_like(self.embedding.weights.data)
        self.projection.weights.grad = np.zeros_like(self.projection.weights.data)
        for block in self.layers:
            for head in block.attention.attn_heads:
                head.weight_Q.grad = np.zeros_like(head.weight_Q.data)
                head.weight_V.grad = np.zeros_like(head.weight_V.data)
                head.weight_K.grad = np.zeros_like(head.weight_K.data)
            for layer, _ in block.feedforward.model.layers:
                layer.weights.grad = np.zeros_like(layer.weights.data)
                layer.biases.grad = np.zeros_like(layer.biases.data)
            block.layernorm1.gamma.grad = np.zeros_like(block.layernorm1.gamma.data)
            block.layernorm1.beta.grad = np.zeros_like(block.layernorm1.beta.data)
            block.layernorm2.gamma.grad = np.zeros_like(block.layernorm2.gamma.data)
            block.layernorm2.beta.grad = np.zeros_like(block.layernorm2.beta.data)