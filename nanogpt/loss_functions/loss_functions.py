import numpy as np
from reversegrad import Tensor
class CrossEntropyLoss:
    def forward(self, predictions, targets):
        self.targets = targets
        self.predictions = predictions
        self.targets = targets
        self.batch_size = Tensor(float(predictions.data.shape[0]))
        batch_indices = np.arange(predictions.data.shape[0])
        self.loss = (Tensor(np.asarray(0.0)) - predictions[batch_indices, targets].log()).sum() / self.batch_size
        return self.loss