import numpy as np
class Optimizer:
    def __init__(self,model , max_iter=100):
        self.max_iter = max_iter
        self.model = model
    def step(self, lr, max_norm=2.5):
        grad_norms=[]
        for w in self.model.weights:
            grad_norms.append(np.linalg.norm(w.grad))
        grad_norm = np.sqrt(np.sum(np.asarray(grad_norms)**2))
        for weight in self.model.weights:
            if grad_norm > max_norm:
                weight.grad = weight.grad/(grad_norm/max_norm)
            weight.data = weight.data-lr*weight.grad
    def zero_grad(self):
        for weight in self.model.weights:
            weight.grad=np.zeros_like(weight.grad)