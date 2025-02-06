import numpy as np
import torch
from networkx import union
from shapely import Polygon
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.SlideSeperatedImageDataset import SlideSeperatedImageDataset
from labelers.GroundTruthLabeler import GroundTruthLabeler
from models.resnet import Resnet18BinaryClassifier
from utils import divide
from pathlib import Path

#%%
slides_root_dir = "data/whole-slides/gut"
annotations_root_dir = "data/annotations/json"
candidates_dataset_dir = "output/candidates"
model_output_dir = "output/models"
#%%
data_split_dict = torch.load(f"{model_output_dir}/data-split.pickle")
model = Resnet18BinaryClassifier(model=torch.load(f"{model_output_dir}/model.pickle"))
train_slides = data_split_dict["train_slides"]
test_slides = data_split_dict["test_slides"]
print("Test slides:", test_slides)
#%%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = model.to(device)
batch_size = 256
test_dataset = SlideSeperatedImageDataset(candidates_dataset_dir, test_slides, with_index=True)
# test_dataset = reduce_dataset(test_dataset, discard_ratio=0)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False, )

print(f"Candidates: {len(test_dataset):,}")
#%%

model.eval()
indexes = []
predictions = []
with torch.no_grad():
    for i, (x_test, y_test, index) in enumerate(tqdm(iter(test_loader), desc=f"Testing")):
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        test_logits = model.forward(x_test)
        test_loss = model.loss_function(test_logits, y_test)
        test_preds = model.predict(test_logits)
        indexes.append(index)
        predictions.append(test_preds.squeeze())
indexes = torch.cat(indexes).to("cpu")
predictions = torch.cat(predictions).to("cpu")
predicted_positives = indexes[predictions == 1]
