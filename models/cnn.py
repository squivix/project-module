import torch
from torch import nn
import torch.nn.functional as F


class CnnModel(nn.Module):
    def __init__(self, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.AdaptiveAvgPool2d((6, 6)),

            nn.Flatten(start_dim=1),

            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        return self.model.forward(x)

    def loss_function(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), target.float())

    def predict(self, logits):
        with torch.no_grad():
            return (torch.sigmoid(logits) > 0.5).float()
