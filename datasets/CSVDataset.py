import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, file_path, feature_start_index=2,skip_header=True, with_index=False):

        self.csv_file_path = file_path

        cached_file_path = f'{"".join(file_path.split(".")[:-1])}.pickle'
        self.with_index = with_index
        if os.path.exists(cached_file_path):
            self.file_paths, self.features, self.labels = torch.load(cached_file_path)
        else:
            file = open(self.csv_file_path, 'r')
            self.features = []
            self.labels = []
            self.file_paths = []
            for i, line in enumerate(file):
                line = line.strip()
                if skip_header and i == 0 or line == "":
                    continue
                tokens = line.split(",")
                self.file_paths.append(tokens[0])
                self.features.append(torch.tensor([float(f) for f in tokens[feature_start_index:-1]]))
                self.labels.append(float(tokens[-1]))
            file.close()
            self.features = torch.stack(self.features)
            self.labels = torch.tensor(self.labels, requires_grad=False)
            torch.save([self.file_paths, self.features, self.labels], cached_file_path)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.with_index:
            return x, y, idx
        else:
            return x, y

    def __len__(self):
        return len(self.labels)
