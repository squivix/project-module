import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


class LabeledImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        if transform is None:
            transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def to_dict(self):
        return {
            "samples": self.samples
        }

    @staticmethod
    def from_dict(dictionary):
        return LabeledImageDataset(dictionary["samples"])
