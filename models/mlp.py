from itertools import chain, repeat

import torch
import torch.nn.functional as F
from torch import nn


class MLPModel(nn.Module):
    def __init__(self, in_features, hidden_layers, units_per_layer, dropout=0.2, threshold=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            *[nn.Linear(in_features, units_per_layer),
              nn.ReLU(),
              nn.Dropout(dropout), ],
            *chain(*repeat(
                [
                    nn.Linear(units_per_layer, units_per_layer),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ], hidden_layers)),
            nn.Linear(units_per_layer, 1),
            nn.Sigmoid()
        )
        self.threshold = threshold

    def forward(self, x):
        return self.model(x)

    def loss_function(self, output, target):
        return F.binary_cross_entropy(output.squeeze(1), target.float())

    def predict(self, prob):
        with torch.no_grad():
            return (prob >= self.threshold).T.float()


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
