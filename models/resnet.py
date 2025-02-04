import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights, ResNet18_Weights

from models.mlp import TransferMLPBinaryClassifier


class Resnet50BinaryClassifier(TransferMLPBinaryClassifier):
    pretrained_output_size = 2048

    @staticmethod
    def create_pretrained_model():
        pretrained_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        pretrained_model.fc = nn.Identity()
        for param in pretrained_model.parameters():
            param.requires_grad = False
        return pretrained_model

    @staticmethod
    def get_pretrained_model_name():
        return 'Resnet50'


class Resnet18BinaryClassifier(TransferMLPBinaryClassifier):
    pretrained_output_size = 512
    @staticmethod
    def create_pretrained_model():
        pretrained_model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        pretrained_model.fc = nn.Identity()
        for param in pretrained_model.parameters():
            param.requires_grad = False
        return pretrained_model

    @staticmethod
    def get_pretrained_model_name():
        return 'Resnet18'
