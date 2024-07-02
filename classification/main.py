import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights

from train import mlp_train
from utils import plot_model_metrics

data_dir = 'data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    transforms.ToTensor(),  # Convert the image to a tensor
])
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
batch_size = 32
train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)
model, model_metrics = mlp_train(model, train_loader, test_dataset, device,
                                 learning_rate=0.001,
                                 max_epochs=10)
plot_model_metrics(model_metrics)
