import json
import os
from pathlib import Path

import pandas as pd

annotations_dir = "data/annotations/json"
slide_df = {"file_name": [], "n_annotations": []}

for annotations_file_name in os.listdir(annotations_dir):
    with open(f"{annotations_dir}/{annotations_file_name}") as f:
        annotations = json.load(f)
        slide_filename = Path(annotations_file_name).stem
        slide_df["file_name"].append(slide_filename)
        slide_df["n_annotations"].append(len(annotations))
slides_df = pd.DataFrame(slide_df)

q1 = slides_df['n_annotations'].quantile(0.25)
median = slides_df['n_annotations'].median()
q3 = slides_df['n_annotations'].quantile(0.75)


# Define new categories based on quartiles
def categorize_quartiles(positive_regions):
    if positive_regions <= q1:
        return "Low"
    elif q1 < positive_regions <= median:
        return "Medium"
    elif median < positive_regions <= q3:
        return "High"
    else:
        return "Very High"


slides_df['category'] = slides_df['n_annotations'].apply(categorize_quartiles)
print(len(slides_df))
train_set_quartiles = pd.DataFrame()
test_set_quartiles = pd.DataFrame()

for category in slides_df['category'].unique():
    category_slides = slides_df[slides_df['category'] == category]
    train = category_slides.sample(frac=0.7, random_state=42)
    test = category_slides.drop(train.index)
    train_set_quartiles = pd.concat([train_set_quartiles, train])
    test_set_quartiles = pd.concat([test_set_quartiles, test])

print(train_set_quartiles)
print()
print(test_set_quartiles)