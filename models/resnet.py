import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights, ResNet18_Weights

from models.mlp import MLPBinaryClassifier


class Resnet50Model(nn.Module):
    def __init__(self, hidden_layers, units_per_layer, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.pretrained_model.fc = nn.Identity()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.model = MLPBinaryClassifier(in_features=2048,
                                         hidden_layers=hidden_layers,
                                         units_per_layer=units_per_layer,
                                         dropout=dropout)

    def forward(self, x):
        pre_logits = self.pretrained_model.forward(x)
        return self.model.forward(pre_logits)

    def loss_function(self, logits, target):
        return self.model.loss_function(logits, target)

    def predict(self, probs):
        return self.model.predict(probs)


class Resnet18Model(nn.Module):
    def __init__(self, hidden_layers, units_per_layer, dropout=0.2, positive_weight=1, negative_weight=1, focal_alpha=0.25, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.pretrained_model.fc = nn.Identity()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.model = MLPBinaryClassifier(in_features=512,
                                         hidden_layers=hidden_layers,
                                         units_per_layer=units_per_layer,
                                         dropout=dropout,
                                         positive_weight=positive_weight,
                                         negative_weight=negative_weight,
                                         focal_alpha=focal_alpha,
                                         focal_gamma=focal_gamma)

    def forward(self, x):
        pre_logits = self.pretrained_model.forward(x)
        return self.model.forward(pre_logits)

    def loss_function(self, logits, target):
        return self.model.loss_function(logits, target)

    def predict(self, probs):
        return self.model.predict(probs)
