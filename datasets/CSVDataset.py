import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, file_path, skip_header=True):

        self.file_path = file_path

        cached_file_path = f'{"".join(file_path.split(".")[:-1])}.pickle'
        print(cached_file_path)
        if os.path.exists(cached_file_path):
            self.features, self.labels = torch.load(cached_file_path)
        else:
            file = open(self.file_path, 'r')
            self.features = []
            self.labels = []
            for i, line in enumerate(file):
                if skip_header and i == 0:
                    continue
                self.features.append(torch.tensor([float(f) for f in line.split(",")[:-2]]))
                self.labels.append(float(line.split(",")[-2]))
            file.close()
            self.features = torch.stack(self.features)
            self.labels = torch.tensor(self.labels, requires_grad=False)
            torch.save([self.features, self.labels], cached_file_path)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.labels)
