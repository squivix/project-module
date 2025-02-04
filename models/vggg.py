import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

from models.mlp import TransferMLPBinaryClassifier


class VGGBinaryClassifier(TransferMLPBinaryClassifier):
    pretrained_output_size = 25088
    @staticmethod
    def create_pretrained_model():
        pretrained_model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        pretrained_model.classifier = nn.Identity()
        for param in pretrained_model.parameters():
            param.requires_grad = False

        return pretrained_model

    @staticmethod
    def get_pretrained_model_name():
        return 'VGG'
