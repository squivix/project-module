import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights


class Resnet50Model(nn.Module):
    def __init__(self, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 1, bias=True)

    def forward(self, x):
        return  self.model.forward(x)

    def loss_function(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), target.float())

    def predict(self, logits):
        with torch.no_grad():
            return (torch.sigmoid(logits) > 0.5).float()
