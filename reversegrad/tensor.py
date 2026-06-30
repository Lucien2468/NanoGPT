import cupy as cp  # was: import numpy as np — swap entire library to GPU arrays

def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape)):
        if shape[i] == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data):
        self.data = cp.asarray(data, dtype=float)  # np.asarray → cp.asarray: puts array on GPU
        self.grad = cp.zeros_like(self.data)        # np.zeros_like → cp.zeros_like: grad lives on GPU too
        self._backward = lambda: None
        self._children = []

    def __add__(self, other):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        out = Tensor(self.data + other.data)
        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        out._children = [self, other]
        return out

    def __sub__(self, other):
        out = Tensor(self.data - other.data)
        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad -= _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        out._children = [self, other]
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data)
        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        out._children = [self, other]
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)
        def _backward():
            self.grad += out.grad @ cp.swapaxes(other.data, -1, -2)  # np.swapaxes → cp.swapaxes
            other_grad = cp.swapaxes(self.data, -1, -2) @ out.grad   # np.swapaxes → cp.swapaxes
            other.grad += cp.sum(other_grad, axis=0) if other_grad.ndim > other.data.ndim else other_grad  # np.sum → cp.sum
        out._backward = _backward
        out._children = [self, other]
        return out

    def __pow__(self, other):
        out = Tensor(self.data ** other.data)
        def _backward():
            self.grad += out.grad * other.data * self.data ** (other.data - 1)
            other.grad += _unbroadcast(out.grad * cp.log(self.data) * self.data ** other.data, other.grad.shape)  # np.log → cp.log
        out._backward = _backward
        out._children = [self, other]
        return out

    def clip(self, min, max):
        out = Tensor(cp.clip(self.data, min, max))  # np.clip → cp.clip
        def _backward():
            self.grad += out.grad * cp.where(out.data == self.data, 1, 0)  # np.where → cp.where
        out._backward = _backward
        out._children = [self]
        return out

    def min(self, other):
        out = Tensor(cp.minimum(self.data, other.data))  # np.minimum → cp.minimum
        def _backward():
            self.grad += out.grad * cp.where(out.data == self.data, 1, 0)  # np.where → cp.where
            other.grad += out.grad * cp.where(out.data == self.data, 0, 1)  # np.where → cp.where
        out._backward = _backward
        out._children = [self, other]
        return out

    def exp(self):
        out = Tensor(cp.exp(self.data))  # np.exp → cp.exp
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        out._children = [self]
        return out

    def __truediv__(self, other):
        out = Tensor(self.data / other.data)
        def _backward():
            other_data = cp.asarray(other.data)  # np.asarray → cp.asarray
            self.grad += out.grad / other_data
            other.grad -= out.grad * self.data / other_data**2 if other_data.shape == () else _unbroadcast(out.grad * self.data / other_data**2, other_data.shape)
        out._backward = _backward
        out._children = [self, other]
        return out

    def log(self):
        out = Tensor(cp.log(self.data))  # np.log → cp.log
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        out._children = [self]
        return out

    def softmax(self):
        exps = cp.exp(self.data - cp.max(self.data, axis=-1, keepdims=True))  # np.exp/np.max → cp.exp/cp.max
        out = Tensor(exps / cp.sum(exps, axis=-1, keepdims=True))             # np.sum → cp.sum
        def _backward():
            s = out.data
            g = out.grad
            dot = cp.sum(s * g, axis=-1, keepdims=True)  # np.sum → cp.sum
            self.grad += s * (g - dot)
        out._backward = _backward
        out._children = [self]
        return out

    def transpose(self):
        out = Tensor(self.data.T)
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        out._children = [self]
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(cp.mean(self.data, axis=axis, keepdims=keepdims))  # np.mean → cp.mean
        def _backward():
            n = self.data.shape[axis] if axis is not None else self.data.size
            self.grad += out.grad * cp.ones_like(self.data) / n  # np.ones_like → cp.ones_like
        out._backward = _backward
        out._children = [self]
        return out

    def var(self, axis=None, keepdims=False):
        out = Tensor(cp.var(self.data, axis=axis, keepdims=keepdims))  # np.var → cp.var
        def _backward():
            n = self.data.shape[axis] if axis is not None else self.data.size
            mean = cp.mean(self.data, axis=axis, keepdims=True)  # np.mean → cp.mean
            grad = out.grad if (keepdims or axis is None) else cp.expand_dims(out.grad, axis=axis)  # np.expand_dims → cp.expand_dims
            self.grad += grad * 2 * (self.data - mean) / n
        out._backward = _backward
        out._children = [self]
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape))
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        out._children = [self]
        return out

    def concat(self, *tensors, axis=0):
        data = [self.data] + [t.data if isinstance(t, Tensor) else t for t in tensors]
        out = Tensor(cp.concatenate(data, axis=axis))  # np.concatenate → cp.concatenate
        def _backward():
            start = 0
            for t in [self] + [t for t in tensors if isinstance(t, Tensor)]:
                end = start + t.data.shape[axis]
                t.grad += out.grad.take(range(start, end), axis=axis)
                start = end
        out._backward = _backward
        out._children = [self] + [t for t in tensors if isinstance(t, Tensor)]
        return out

    def append(self, other, axis=None):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_data = cp.append(self.data, other.data, axis=axis)  # np.append → cp.append
        out = Tensor(new_data)
        def _backward():
            if axis is None:
                self_size = self.data.size
                self_grad = out.grad[:self_size]
                self_grad = self_grad.reshape(self.data.shape)
                self.grad += self_grad
                other_grad = out.grad[self_size:]
                other_grad = other_grad.reshape(other.data.shape)
                other.grad += other_grad
            else:
                self_size = self.data.shape[axis]
                out_size = out.data.shape[axis]
                self_indices = list(range(0, self_size))
                other_indices = list(range(self_size, out_size))
                self.grad += cp.take(out.grad, self_indices, axis=axis)   # np.take → cp.take
                other.grad += cp.take(out.grad, other_indices, axis=axis) # np.take → cp.take
        out._backward = _backward
        out._children = [self, other]
        return out

    def swapaxes(self, axis_1, axis_2):
        new_data = cp.swapaxes(self.data, axis_1, axis_2)  # np.swapaxes → cp.swapaxes
        out = Tensor(new_data)
        def _backward():
            self.grad += cp.swapaxes(out.grad, axis_1, axis_2)  # np.swapaxes → cp.swapaxes
        out._backward = _backward
        out._children = [self]
        return out

    def __getitem__(self, idx):
        idx = idx.data if isinstance(idx, Tensor) else idx
        out = Tensor(self.data[idx])
        def _backward():
            if not isinstance(self.grad, cp.ndarray):  # np.ndarray → cp.ndarray
                self.grad = cp.zeros_like(self.data)   # np.zeros_like → cp.zeros_like
            self.grad[idx] += out.grad
        out._backward = _backward
        out._children = [self]
        return out

    def sum(self):
        out = Tensor(cp.asarray(cp.sum(self.data)))  # np.asarray/np.sum → cp.asarray/cp.sum
        def _backward():
            self.grad += out.grad
        out._backward = _backward
        out._children = [self]
        return out

    def backward(self, grad=1, clear_children=False):
        self.grad += grad
        topo = []
        visited = list()

        def build_topo(node):
            if id(node) not in visited:
                visited.append(id(node))
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        for i, node in enumerate(reversed(topo)):
            node._backward()
        if clear_children:
            node._children = []
