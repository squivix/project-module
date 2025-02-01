import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.SlideSeperatedDataset import SlideSeperatedDataset
from models.resnet import Resnet50Model
from utils import reduce_dataset, split_dataset

slides_root_dir = "data/whole-slides/gut"
annotations_root_dir = "data/annotations/json"
candidates_dataset_dir = "output/candidates"


def load_annotations(directory):
    slide_data = {"slide_name": [], "n_positive_annotations": []}
    for annotations_file in os.listdir(directory):
        with open(f"{directory}/{annotations_file}") as f:
            annotations = json.load(f)
            slide_name = Path(annotations_file).stem
            slide_data["slide_name"].append(slide_name)
            slide_data["n_positive_annotations"].append(len(annotations))
    return pd.DataFrame(slide_data)


def assign_categories(dataframe):
    q1, median, q3 = dataframe['n_positive_annotations'].quantile([0.25, 0.5, 0.75])

    def categorize_quartiles(n_annotations):
        if n_annotations <= q1:
            return "Low"
        elif q1 < n_annotations <= median:
            return "Medium"
        elif median < n_annotations <= q3:
            return "High"
        else:
            return "Very High"

    dataframe['category'] = dataframe['n_positive_annotations'].apply(lambda x: categorize_quartiles(x))
    return dataframe


def split_data(dataframe, train_portion=0.7):
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for category in dataframe['category'].unique():
        category_slides = dataframe[dataframe['category'] == category]
        train_samples = category_slides.sample(frac=train_portion)
        test_samples = category_slides.drop(train_samples.index)
        train_set = pd.concat([train_set, train_samples])
        test_set = pd.concat([test_set, test_samples])
    return train_set, test_set


slides_df = load_annotations(annotations_root_dir)
slides_df = assign_categories(slides_df)
train_slides, test_slides = split_data(slides_df)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

batch_size = 256
dataset = SlideSeperatedDataset(candidates_dataset_dir, set(train_slides["slide_name"]))


dataset = reduce_dataset(dataset, discard_ratio=0.0)
train_dataset, validation_dataset = split_dataset(dataset, train_ratio=0.7)
# train_dataset = undersample_dataset(train_dataset)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(validation_dataset,
                         batch_size=batch_size,
                         shuffle=True, )

model = Resnet50Model(hidden_layers=1, units_per_layer=2048,
                      dropout=0.3, focal_alpha=0.9, focal_gamma=2.0)

print(f"Dataset: {len(train_dataset):,} training, {len(validation_dataset):,} validation")
