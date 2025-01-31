import os

import matplotlib
from torch.utils.data import Dataset

from datasets.default_image_transform import default_image_transform

matplotlib.use('qtagg')

if hasattr(os, 'add_dll_directory'):
    # Windows
    OPENSLIDE_PATH = os.path.join(os.path.abspath(os.getcwd()),
                                  "libs/openslide-bin-4.0.0.3-windows-x64/bin")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


class SlideDataset(Dataset):
    def __init__(self, slide_file_paths, extract_candidates, transform=None):
        if transform is None:
            transform = default_image_transform
        self.candidates = []
        for slide_file_path in slide_file_paths:
            slide = openslide.OpenSlide(slide_file_path)
            self.candidates.extend(extract_candidates(slide))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
