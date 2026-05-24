
import numpy as np
from reversegrad import Tensor
class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.embed_dim = embed_dim
        self.eps = np.asarray(eps)
        self.gamma = Tensor(np.ones((1, embed_dim)))
        self.beta = Tensor(np.zeros((1, embed_dim)))
    def forward(self, x):
        x = x if hasattr(x, 'data') else Tensor(x)
        self.x = x
        mean = x.mean(axis=-1, keepdims=True)

        var = x.var(axis=-1, keepdims=True)
        normalized_numerator = x - mean
        variance_with_eps = var + self.eps
        std = variance_with_eps ** Tensor(0.5)
        self.x_normalized = normalized_numerator / std
        return self.gamma * self.x_normalized + self.beta
'''
    def backward(self, grad_output):
        n = self.x_normalized.data.shape[-1]
        std = np.sqrt(np.var(self.x.data, axis=-1, keepdims=True) + self.eps.data)

        self.gamma_grad = np.sum(grad_output * self.x_normalized.data, axis=0)
        self.beta_grad = np.sum(grad_output, axis=0)
        self.x_grad = (1.0 / (n * std)) * (n * grad_output * self.gamma.data - np.sum(grad_output * self.gamma.data, axis=-1, keepdims=True) - self.x_normalized.data * np.sum(grad_output * self.gamma.data * self.x_normalized.data, axis=-1, keepdims=True))

import numpy as np
class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = np.ones((embed_dim,))
        self.beta = np.zeros((embed_dim,))
    def forward(self, x):
        self.x = x
        mean = x.mean(axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.x_normalized = (x - mean) / (var + np.full(var.shape, self.eps)) ** np.full(var.shape, 1/2)
        return self.gamma * self.x_normalized + self.beta

    def backward(self, grad_output):
        mean = self.x_normalized.mean(axis=-1, keepdims=True)
        var = np.var(self.x, axis=-1, keepdims=True)
        n = self.x_normalized.shape[-1]
        std = (np.sqrt(np.var(self.x, axis=-1, keepdims=True) + self.eps))

        self.gamma_grad = np.sum(grad_output * self.x_normalized, axis=0)
        self.beta_grad = np.sum(grad_output, axis=0)
        self.x_grad = (1.0 / (n * std)) * (n * grad_output * self.gamma - np.sum(grad_output * self.gamma, axis=-1, keepdims=True) - self.x_normalized * np.sum(grad_output * self.gamma * self.x_normalized, axis=-1, keepdims=True))
'''

def grad_check(x, theta, epsilon=1e-4):
    ln = LayerNorm(theta.shape[0])
    ln.gamma = theta
    ln.beta = np.zeros_like(theta)
    ln.x_grad = np.zeros_like(x)
    out = ln.forward(x)
    grad_output = np.random.randn(*out.shape)
    ln.backward(grad_output)
    numerical_gamma_grad = np.zeros_like(theta)
    numerical_beta_grad = np.zeros_like(theta)
    numerical_x_grad = np.zeros_like(x)
    loss = np.sum(out ** 2)

    for i in range(len(theta)):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        ln.gamma = theta_plus
        out_plus = ln.forward(x)
        ln.gamma = theta_minus
        out_minus = ln.forward(x)

        numerical_gamma_grad[i] = (np.sum(out_plus * grad_output) - np.sum(out_minus * grad_output)) / (2 * epsilon)
    for i in range(len(theta)):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        ln.beta = theta_plus
        out_plus = ln.forward(x)
        ln.beta = theta_minus
        out_minus = ln.forward(x)

        numerical_beta_grad[i] = (np.sum(out_plus * grad_output) - np.sum(out_minus * grad_output)) / (2 * epsilon)

    for idx in np.ndindex(x.shape):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[idx] += epsilon
        x_minus[idx] -= epsilon
        out_plus = ln.forward(x_plus)
        out_minus = ln.forward(x_minus)
        numerical_x_grad[idx] = (np.sum(out_plus * grad_output) - np.sum(out_minus * grad_output)) / (2 * epsilon)

    print("Analytical gamma grad:", ln.gamma_grad)
    print("Numerical gamma grad:", numerical_gamma_grad)
    print("Analytical beta grad:", ln.beta_grad)
    print("Numerical beta grad:", numerical_beta_grad)
    print("Analytical x grad:", ln.x_grad)
    print("Numerical x grad:", numerical_x_grad)    