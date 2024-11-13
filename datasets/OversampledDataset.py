from torch.utils.data import Dataset


class OversampledDataset(Dataset):
    def __init__(self, original_dataset, oversampled_indexes, transform):
        self.original_dataset = original_dataset
        self.oversampled_indexes = oversampled_indexes
        # self.labels
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset) + len(self.oversampled_indexes)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]

        original_idx = self.oversampled_indexes[idx - len(self.original_dataset)]
        x, y = self.original_dataset.get_item_untransformed(original_idx)
        if self.transform is not None:
            x = self.transform(x)
        return x, y
