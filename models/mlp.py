import torch
from torch import nn

import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, in_features, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        return self.model(x)

    def loss_function(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), target.float())

    def predict(self, logits):
        with torch.no_grad():
            return (torch.sigmoid(logits) > 0.5).float()
