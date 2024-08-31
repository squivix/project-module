import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models import VGG16_Weights


class VggModel(nn.Module):
    def __init__(self, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
        )

    def forward(self, x):
        pre_logits = self.pretrained_model.forward(x)
        return self.model.forward(pre_logits)

    def loss_function(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), target.float())

    def predict(self, logits):
        with torch.no_grad():
            return (torch.sigmoid(logits) > 0.5).float()
