import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from candidate_extractors.TemplateMatchExtractor import TemplateMatchExtractor
from candidate_extractors.TemplateMatchExtractor import generate_dataset_from_slides
from datasets.SlideSeperatedCSVDataset import SlideSeperatedCSVDataset
from models.mlp import MLPBinaryClassifier
from models.resnet import Resnet18BinaryClassifier
from train import train_classifier
from utils import extract_features_from_dataset
from utils import plot_model_metrics
from utils import reduce_dataset, split_dataset
from datasets.SlideSeperatedImageDataset import SlideSeperatedImageDataset

#%%
slides_root_dir = "data/whole-slides/gut"
annotations_root_dir = "data/annotations/json"
candidates_dataset_dir = "output/candidates"
model_output_dir = "output/models"
PretrainedModelClass = Resnet18BinaryClassifier
features_csv_file_name = f"{PretrainedModelClass.get_pretrained_model_name()}_{PretrainedModelClass.pretrained_output_size}_features.csv"
print(f"{PretrainedModelClass.get_pretrained_model_name()}: {PretrainedModelClass.pretrained_output_size} features")
#%%
extractor = TemplateMatchExtractor()
generate_dataset_from_slides(slides_root_dir, extractor, candidates_dataset_dir)
#%%
extract_features_from_dataset(candidates_dataset_dir,
                              [Resnet18BinaryClassifier])
#%%

#%%
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
print("Train Slides")
train_slides
#%%
print("Test Slides")
test_slides
#%%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

batch_size = 256
# dataset = SlideSeperatedImageDataset(candidates_dataset_dir, set(train_slides["slide_name"]))
dataset = SlideSeperatedCSVDataset(f"{candidates_dataset_dir}/{features_csv_file_name}", set(train_slides["slide_name"]))
dataset = reduce_dataset(dataset, discard_ratio=0.0)
train_dataset, validation_dataset = split_dataset(dataset, train_ratio=0.9)
# train_dataset = undersample_dataset(train_dataset)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
validation_loader = DataLoader(validation_dataset,
                               batch_size=batch_size,
                               shuffle=True, )

# model = Resnet18BinaryClassifier(hidden_layers=1, units_per_layer=2048,
#                       dropout=0.3, focal_alpha=0.9, focal_gamma=2.0)
model = MLPBinaryClassifier(in_features=PretrainedModelClass.pretrained_output_size, hidden_layers=1,
                            units_per_layer=PretrainedModelClass.pretrained_output_size,
                            dropout=0.3, focal_alpha=0.9, focal_gamma=2.0)

print(f"Dataset: {len(train_dataset):,} training, {len(validation_dataset):,} validation")