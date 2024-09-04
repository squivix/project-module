import os

import torch
from torch.utils.data import Dataset


class CSVStreamDataset(Dataset):
    def __init__(self, file_path, skip_header=True):
        cached_offsets_file_path = f'{"".join(file_path.split(".")[:-1])}-offsets.pickle'
        self.file_path = file_path
        offsets = []
        labels = []
        if os.path.exists(cached_offsets_file_path):
            self.offsets, self.labels = torch.load(cached_offsets_file_path)
        else:
            with open(file_path, 'r') as csv_file:
                offset = csv_file.tell()
                i = 0
                line = csv_file.readline().strip()
                while line:
                    if i == 0 and skip_header:
                        line = csv_file.readline().strip()
                        i += 1
                        continue
                    label = int(line.split(",")[-2])
                    offsets.append(offset)
                    labels.append(label)
                    offset = csv_file.tell()
                    line = csv_file.readline().strip()
                    i += 1
            self.offsets = torch.tensor(offsets, requires_grad=False)
            self.labels = torch.tensor(labels, requires_grad=False)
            torch.save([self.offsets, self.labels], cached_offsets_file_path)

    def __getitem__(self, idx):
        file = open(self.file_path, 'r')

        byte_offset = self.offsets[idx].item()
        file.seek(byte_offset)
        line = file.readline()
        file.close()
        x = torch.tensor([float(f) for f in line.split(",")[:-2]])
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.labels)
