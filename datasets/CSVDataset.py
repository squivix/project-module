from torch.utils.data import Dataset

import torch


class CSVDataset(Dataset):
    def __init__(self, file_path, samples, transform=None):
        self.file_path = file_path
        self.samples = samples

    def __getitem__(self, idx):
        file = open(self.file_path, 'r')

        byte_offset, label = self.samples[idx]
        file.seek(byte_offset)
        line = file.readline()
        file.close()
        x = torch.tensor([float(f) for f in line.split(",")[:-2]])
        return x, label

    def __len__(self):
        return len(self.samples)