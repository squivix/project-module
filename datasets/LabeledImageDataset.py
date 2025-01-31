import glob
import os
import pickle

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

from datasets.default_image_transform import default_image_transform


class LabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, extension='.[jpg][png]*', ignore_cache=False, with_index=False):
        cache_file_path = f"{os.path.dirname(root_dir)}/{os.path.basename(root_dir)}-cache.pickle"

        if os.path.exists(cache_file_path) and not ignore_cache:
            with open(cache_file_path, "rb") as f:
                file_paths, labels = pickle.load(f)
        else:
            file_paths = []
            labels = []
            for class_index, class_name in enumerate(os.listdir(root_dir)):
                class_dir = os.path.join(root_dir, class_name)
                for file_path in sorted([path for path in glob.glob(f"{class_dir}/*{extension}")]):
                    file_paths.append(file_path)
                    labels.append(class_index)
            pickle.dump([file_paths, labels], open(cache_file_path, "wb"))

        self.file_paths = file_paths
        self.labels = torch.tensor(labels, requires_grad=False)

        if transform is None:
            transform = default_image_transform
        self.transform = transform
        self.with_index = with_index

    def __len__(self):
        return len(self.labels)

    def get_item_untransformed(self, idx):
        img_path = self.file_paths[idx]
        x = read_image(img_path, mode=ImageReadMode.RGB)
        return x

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        x = read_image(img_path, mode=ImageReadMode.RGB)

        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]
        if self.with_index:
            return x, y, idx
        return x, y

    def to_dict(self):
        return {
            "file_paths": self.file_paths,
            "labels": self.file_paths
        }
