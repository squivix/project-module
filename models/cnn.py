from torch import nn


class CnnModel(nn.Module):
    def __init__(self, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding="valid"),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 16, kernel_size=3, stride=3, padding="valid"),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=3, padding="valid"),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=3, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        return self.model.forward(x)