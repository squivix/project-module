import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class VggModel(nn.Module):
    def __init__(self, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.vgg16()
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=1, bias=True)

    def forward(self, x):
        return self.model.forward(x)

    def loss_function(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), target.float())

    def predict(self, logits):
        with torch.no_grad():
            return (torch.sigmoid(logits) > 0.5).float()
