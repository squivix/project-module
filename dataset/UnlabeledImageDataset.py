import glob
import os.path

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

from utils import rescale_data_transform


# rescale_data_transform(0, 255, -1, +1)

class UnlabeledImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transformations = v2.Compose([
            # Drop the alpha channel

            v2.Lambda(lambda t: t[:3, :, :]),
            # Normalize pixel values to be -1 to 1
            v2.Lambda(lambda t: (t / 255) * 2 - 1),
        ])
        if not os.path.isdir(img_dir):
            raise ValueError('Image directory does not exist')
        self.image_file_names = sorted([os.path.basename(path) for path in glob.glob(f"{img_dir}/*.png")])
        self.classes = ["Negative", "Positive"]

    def __len__(self):
        return len(self.image_file_names)

    def get_raw_image(self, idx):
        img_path = os.path.join(self.img_dir, self.image_file_names[idx])
        img = read_image(img_path)
        return img[:3, :, :]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_file_names[idx])
        img = read_image(img_path)

        x = self.transformations(img)
        return x
