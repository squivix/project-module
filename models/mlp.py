from itertools import chain, repeat

import torch
import torch.nn.functional as F
from torch import nn


class MLPBinaryClassifier(nn.Module):
    def __init__(self, in_features, hidden_layers, units_per_layer, dropout=0.2, threshold=0.5, positive_weight=1, negative_weight=1, focal_alpha=0.25, focal_gamma=2.0, *args,
                 **kwargs):
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
        self.hidden_layers = hidden_layers
        self.threshold = threshold
        self.negative_weight = negative_weight
        self.positive_weight = positive_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, x):
        if self.hidden_layers == 0:
            return x
        return self.model(x)

    def loss_function(self, output, target):
        return self.focal_loss_function(output, target)

    def bce_loss_function(self, output, target):
        output = output.squeeze(1)
        return F.binary_cross_entropy(output, target.float(), weight=torch.where(target == 1,
                                                                                 self.positive_weight * torch.ones_like(output),
                                                                                 self.negative_weight * torch.ones_like(output))
                                      )

    def focal_loss_function(self, output, target):
        output = output.squeeze(1)
        target = target.float()

        bce_loss = F.binary_cross_entropy(output, target, reduction='none')

        pt = torch.where(target == 1, output, 1 - output)
        focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma

        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()

    def predict(self, prob):
        with torch.no_grad():
            return (prob >= self.threshold).T.float()


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
