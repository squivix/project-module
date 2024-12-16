import torchvision
from torch import nn
from torchvision.models import Inception_V3_Weights, InceptionOutputs

from models.mlp import MLPBinaryClassifier


class InceptionV3Model(nn.Module):
    def __init__(self, hidden_layers, units_per_layer, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.pretrained_model.AuxLogits = None
        self.pretrained_model.fc = nn.Identity()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.model = MLPBinaryClassifier(in_features=2048,
                                         hidden_layers=hidden_layers,
                                         units_per_layer=units_per_layer,
                                         dropout=dropout)

    def forward(self, x):
        output = self.pretrained_model.forward(x)
        if isinstance(output, InceptionOutputs):
            pre_logits = output.logits
        else:
            pre_logits = output
        return self.model.forward(pre_logits)

    def loss_function(self, logits, target):
        return self.model.loss_function(logits, target)

    def predict(self, probs):
        return self.model.predict(probs)
