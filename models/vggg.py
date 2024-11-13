import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

from models.mlp import MLPModel


class VggModel(nn.Module):
    def __init__(self, hidden_layers, units_per_layer, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.pretrained_model.classifier = nn.Identity()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.model = MLPModel(in_features=25088, hidden_layers=hidden_layers, units_per_layer=units_per_layer, dropout=dropout)

    def forward(self, x):
        pre_logits = self.pretrained_model.forward(x)
        return self.model.forward(pre_logits)

    def loss_function(self, logits, target):
        return self.model.loss_function(logits, target)

    def predict(self, probs):
        return self.model.predict(probs)
