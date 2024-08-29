from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

from utils import rescale_data_transform


class LabeledImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = v2.Compose([
            v2.ToImage(),
            # v2.Resize((256, 256)),
            v2.Resize(299),
            v2.CenterCrop(299),

            rescale_data_transform(0, 255, 0, 1)
        ])

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
