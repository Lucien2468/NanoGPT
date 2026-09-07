import cupy as cp  # was: import numpy as np
from reversegradGPU import Tensor
class CrossEntropyLoss:
    def forward(self, predictions, targets, mask = None):
        self.batch_size = Tensor(float(predictions.data.shape[0])) if predictions.data.ndim == 3 else 1
        self.seq_len = Tensor(float(predictions.data.shape[1 if predictions.data.ndim == 3 else 0]))
        denominator = self.seq_len * self.batch_size
        if isinstance(mask, cp.ndarray):
            denominator = Tensor(float(mask.size - cp.count_nonzero(mask == 0)))
        self.loss = self.forward_seq(predictions, targets, mask).sum() / denominator
        return self.loss
    def forward_seq(self, predictions, targets, mask = None):
        self.targets = targets
        self.predictions = predictions.softmax()
        self.batch_size = Tensor(float(predictions.data.shape[0])) if predictions.data.ndim == 3 else 1
        self.seq_len = Tensor(float(predictions.data.shape[1 if predictions.data.ndim == 3 else 0]))
        seq_indices = cp.arange(predictions.data.shape[1 if predictions.data.ndim == 3 else 0])[None, :]  # np.arange → cp.arange: index array on GPU
        batch_indices = cp.arange(predictions.data.shape[0])[:, None] if predictions.data.ndim == 3 else None  # np.arange → cp.arange
        if batch_indices is None:
            self.loss = Tensor(cp.asarray(0.0)) - self.predictions[seq_indices, targets].log()
        else:
            self.loss = Tensor(cp.asarray(0.0)) - self.predictions[batch_indices, seq_indices, targets].log()
        return self.loss * mask if isinstance(mask, cp.ndarray) else self.loss
def grad_check(loss_fn, pred_data, targets, eps=1e-4, rtol=1e-3):
    pred_data = cp.asarray(pred_data, dtype=float)
    n = pred_data.size

    pred_tensor = Tensor(pred_data.copy())
    loss = loss_fn.forward(pred_tensor, targets)
    loss.backward()
    analytic = pred_tensor.grad.ravel()

    numeric = cp.zeros(n)
    for i in range(n):
        perturbed = pred_data.copy()
        p = perturbed.ravel()

        p[i] += eps
        lp = float(loss_fn.forward(Tensor(perturbed), targets).data)

        p[i] -= 2 * eps
        lm = float(loss_fn.forward(Tensor(perturbed), targets).data)

        numeric[i] = (lp - lm) / (2 * eps)

    rel_err = float(cp.max(cp.abs(analytic - numeric) / (cp.abs(analytic) + cp.abs(numeric) + 1e-8)))
    return rel_err < rtol, rel_err