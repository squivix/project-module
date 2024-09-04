import glob
import os
import pickle

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


class LabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        cache_file_path = f"{os.path.dirname(root_dir)}/{os.path.basename(root_dir)}-cache.pickle"
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "rb") as f:
                file_paths, labels = pickle.load(f)
        else:
            file_paths = []
            labels = []
            for class_index, class_name in enumerate(os.listdir(root_dir)):
                class_dir = os.path.join(root_dir, class_name)
                for file_path in sorted([path for path in glob.glob(f"{class_dir}/*.jpeg")]):
                    file_paths.append(file_path)
                    labels.append(class_index)
            pickle.dump([file_paths, labels], open(cache_file_path, "wb"))

        self.file_paths = file_paths
        self.labels = torch.tensor(labels, requires_grad=False)

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
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        x = read_image(img_path)

        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]
        return x, y

    def to_dict(self):
        return {
            "file_paths": self.file_paths,
            "labels": self.file_paths
        }

    # @staticmethod
    # def from_dict(dictionary):
    #     return LabeledImageDataset(dictionary["samples"])
