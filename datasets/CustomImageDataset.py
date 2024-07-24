import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from utils import rescale_data_transform


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, class_size=10):
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((256, 256)),
            rescale_data_transform(0, 255, -1, +1)
        ])
        self.class_size = class_size
        dataset_class_0 = ImageFolder(root=f"{data_dir}/Negative", transform=self.transform)
        dataset_class_1 = ImageFolder(root=f"{data_dir}/Positive", transform=self.transform,
                                      target_transform=v2.Lambda(lambda x: x + 1))  # hack

        self.dataset = ConcatDataset([
            Subset(dataset_class_0, np.random.choice(len(dataset_class_0), class_size, replace=False)),
            Subset(dataset_class_1, np.random.choice(len(dataset_class_1), class_size, replace=False)),
        ])

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.class_size * 2
