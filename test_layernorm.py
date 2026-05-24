import numpy as np
from nanogpt import LayerNorm
from reversegrad import Tensor

np.random.seed(42)

ln = LayerNorm(8)
x = Tensor(np.random.randn(4, 8))
out = ln.forward(x)

out.backward()
print(out.grad)