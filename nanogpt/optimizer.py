from nanogpt.layers.layernorm import LayerNorm
import numpy as np
class Optimizer:
    def __init__(self,model , max_iter=100):
        self.max_iter = max_iter
        self.model = model
        self.weights_adam = []
        for weight in self.model.weights:
            m = np.zeros_like(weight.data)
            v = np.zeros_like(weight.data)
            self.weights_adam.append((weight, m, v))
        self.time=1
    def step(self, lr,  beta1=0.9, beta2=0.999, max_norm=2.5):
        grad_norms=[]
        for i in range(len(self.weights_adam)):grad_norms.append(np.linalg.norm(self.weights_adam[i][0].grad))
        grad_norm = np.sqrt(np.sum(np.asarray(grad_norms)**2))
        for i in range(len(self.weights_adam)):
            weight = self.weights_adam[i][0]
            if grad_norm > max_norm:
                weight.grad = weight.grad/(grad_norm/max_norm)
            m = beta1*self.weights_adam[i][1] + (1-beta1)*weight.grad
            v = beta2*self.weights_adam[i][2] + (1-beta2)*(weight.grad**2)
            m_hat = m / (1 - beta1**self.time)
            v_hat = v / (1 - beta2**self.time)
            weight.data = weight.data-lr*(m_hat/(v_hat**0.5+1e-7))
            self.weights_adam[i] = (weight, m, v)
        self.time+=1

    def zero_grad(self):
        for weight in self.weights_adam:
            weight[0].grad=np.zeros_like(weight[0].grad)