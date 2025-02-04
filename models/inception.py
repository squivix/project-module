from typing import override

import torchvision
from torch import nn
from torchvision.models import Inception_V3_Weights, InceptionOutputs

from models.mlp import TransferMLPBinaryClassifier


class InceptionV3BinaryClassifier(TransferMLPBinaryClassifier):
    pretrained_output_size = 2048

    @staticmethod
    def create_pretrained_model():
        pretrained_model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        pretrained_model.AuxLogits = None
        pretrained_model.fc = nn.Identity()
        for param in pretrained_model.parameters():
            param.requires_grad = False
        return pretrained_model

    @staticmethod
    def get_pretrained_model_name():
        return 'InceptionV3'

    @override
    def pre_forward(self, x):
        output = self.pre_forward(x)
        if isinstance(output, InceptionOutputs):
            pre_logits = output.logits
        else:
            pre_logits = output
        return pre_logits
