import pandas as pd
import torch
from torch.utils.data import Dataset


class SlideSeperatedCSVDataset(Dataset):
    def __init__(self, csv_path, included_slides_names=None):
        self.data = pd.read_csv(csv_path)

        if included_slides_names is not None:
            self.data = self.data[self.data['slide'].astype("string").isin(included_slides_names)]

        self.file_paths = self.data['file_path'].tolist()

        feature_columns = self.data.columns[2:-1]  # Exclude 'file_name', 'slide', and 'label'
        self.x_data = self.data[feature_columns].values.astype(float)
        self.labels = torch.tensor(self.data['label'].values.astype(float), requires_grad=False)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y = self.labels[idx]
        return x, y

    def get_item_file_path(self, idx):
        return self.file_paths[idx]
