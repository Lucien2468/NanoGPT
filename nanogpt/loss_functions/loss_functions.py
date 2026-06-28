import numpy as np
from reversegrad import Tensor
class CrossEntropyLoss:
    def forward(self, predictions, targets):
        self.targets = targets
        self.predictions = predictions.softmax()
        self.targets = targets
        self.batch_size = Tensor(float(predictions.data.shape[0])) if predictions.data.ndim == 3 else 1
        self.seq_len = Tensor(float(predictions.data.shape[1 if predictions.data.ndim == 3 else 0]))
        seq_indices = np.arange(predictions.data.shape[1 if predictions.data.ndim == 3 else 0])[None, :]
        batch_indices = np.arange(predictions.data.shape[0])[:, None] if predictions.data.ndim == 3 else None
        if batch_indices is None:
            self.loss = (Tensor(np.asarray(0.0)) - predictions[seq_indices, targets].log()).sum() / self.seq_len
        else:
            self.loss = (Tensor(np.asarray(0.0)) - predictions[batch_indices, seq_indices, targets].log()).sum() / (self.batch_size * self.seq_len)
        return self.loss