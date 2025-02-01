import os
from itertools import chain

import pandas as pd
from torch.utils.data import Dataset

from datasets.LabeledImageDataset import LabeledImageDataset


class SlideSeperatedDataset(Dataset):
    def __init__(self, root_dir, included_slides_names, transform=None, extension='.[jpg][png]*', with_index=False):
        self.slide_to_dataset = {}
        self.slides_length_df = {"slide": [], "dataset_length": []}
        self.with_index = with_index
        self.labels = []
        for slide in os.listdir(root_dir):
            slide_dir = os.path.join(root_dir, slide)
            if not os.path.isdir(slide_dir) or slide not in included_slides_names:
                continue
            self.slide_to_dataset[slide] = LabeledImageDataset(slide_dir, transform=transform, extension=extension, with_index=with_index)
            self.labels.extend(self.slide_to_dataset[slide].labels)
            self.slides_length_df["slide"].append(slide)
            self.slides_length_df["dataset_length"].append(len(self.slide_to_dataset[slide]))
        self.slides_length_df = pd.DataFrame(self.slides_length_df)
        self.slides_length_df["dataset_length"] = self.slides_length_df["dataset_length"].astype(int)

    def __len__(self):
        return self.slides_length_df["dataset_length"].sum().item()

    def _flat_index_to_slide_index(self, index):
        cumulative_length = 0

        for _, row in self.slides_length_df.iterrows():
            name, length = row["slide"], row["dataset_length"]
            if index < cumulative_length + length:
                return name, index - cumulative_length
            cumulative_length += length

    def get_item_untransformed(self, idx):
        slide, sub_index = self._flat_index_to_slide_index(idx)
        return self.slide_to_dataset[slide].get_item_untransformed(sub_index)

    def get_item_file_path(self, idx):
        slide, sub_index = self._flat_index_to_slide_index(idx)
        return self.slide_to_dataset[slide].file_paths[sub_index]

    def __getitem__(self, idx):
        slide, sub_index = self._flat_index_to_slide_index(idx)

        if self.with_index:
            x, y, _ = self.slide_to_dataset[slide][sub_index]
            return x, y, idx
        else:
            return self.slide_to_dataset[slide][sub_index]
