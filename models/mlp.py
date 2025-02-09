import abc
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
        self.default_threshold = threshold
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

        pt = output * target + (1 - output) * (1 - target)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()

    def predict(self, prob, threshold=None):
        if threshold is None:
            threshold = self.default_threshold
        with torch.no_grad():
            return (prob >= threshold).T.float()


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class TransferMLPBinaryClassifier(nn.Module, abc.ABC):
    def __init__(self, hidden_layers=1, model=None, units_per_layer=128, dropout=0.2, positive_weight=1, negative_weight=1, focal_alpha=0.25, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pretrained_model = self.create_pretrained_model()
        if model is not None:
            self.model = model
        else:
            self.model = MLPBinaryClassifier(in_features=self.pretrained_output_size,
                                             hidden_layers=hidden_layers,
                                             units_per_layer=units_per_layer,
                                             dropout=dropout,
                                             positive_weight=positive_weight,
                                             negative_weight=negative_weight,
                                             focal_alpha=focal_alpha,
                                             focal_gamma=focal_gamma)

    def forward(self, x):
        pre_logits = self.pre_forward(x)
        return self.model.forward(pre_logits)

    @staticmethod
    @abc.abstractmethod
    def create_pretrained_model():
        pass

    @staticmethod
    @abc.abstractmethod
    def get_pretrained_model_name():
        pass

    def pre_forward(self, x):
        with torch.no_grad():
            return self.pretrained_model.eval().forward(x)

    def loss_function(self, logits, target):
        return self.model.loss_function(logits, target)

    def predict(self, probs,threshold=None):
        return self.model.predict(probs, threshold)
