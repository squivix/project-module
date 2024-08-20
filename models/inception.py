import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models import Inception_V3_Weights, InceptionOutputs


class InceptionV3Model(nn.Module):
    def __init__(self, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.AuxLogits = None
        self.model.fc = nn.Linear(2048, 1)

    def forward(self, x):
        output = self.model.forward(x)
        if isinstance(output, InceptionOutputs):
            return output.logits
        else:
            return output

    def loss_function(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), target.float())

    def predict(self, logits):
        with torch.no_grad():
            return (torch.sigmoid(logits) > 0.5).float()
