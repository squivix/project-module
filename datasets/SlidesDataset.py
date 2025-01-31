import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.default_image_transform import default_image_transform

if hasattr(os, 'add_dll_directory'):
    # Windows
    OPENSLIDE_PATH = os.path.join(os.path.abspath(os.getcwd()), "libs/openslide-bin-4.0.0.3-windows-x64/bin")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

# Too slow :(
class SlideDataset(Dataset):
    def __init__(self, slides_dir, extractor, cache_dir="data", transform=None):
        if transform is None:
            transform = default_image_transform
        self.transform = transform
        self.slides_dir = slides_dir

        cache_filepath = os.path.join(cache_dir, f"slide-dataset-cache.pickle")
        if os.path.exists(cache_filepath):
            with open(cache_filepath, "rb") as f:
                self.slide_candidates = pickle.load(f)
        else:
            self.slide_candidates = {}

            for slide_filename in os.listdir(slides_dir):
                if Path(slide_filename).suffix != ".svs":
                    continue
                slide_filepath = os.path.join(slides_dir, slide_filename)
                self.slide_candidates[slide_filename] = extractor.extract_candidates(slide_filepath)
            with open(cache_filepath, "wb") as f:
                pickle.dump(self.slide_candidates, f)

        self.candidates = []
        self.labels = []
        for slide_filename in self.slide_candidates.keys():
            for i in range(len(self.slide_candidates[slide_filename])):
                self.candidates.append((slide_filename, i))
                self.labels.append(1 if self.slide_candidates[slide_filename][i]["is_positive"] else 0)

    def __getitem__(self, idx):
        slide_name, candidate_index = self.candidates[idx]
        candidate_bbox = self.slide_candidates[slide_name][candidate_index]["candidate_bbox"]
        slide = openslide.OpenSlide(os.path.join(self.slides_dir, slide_name))
        x, y, w, h = candidate_bbox
        x = np.array(slide.read_region((x, y), 0, (w, h)))
        slide.close()
        x = self.transform(torch.tensor(x[:, :, :3]).transpose(-1, 0))
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.candidates)
