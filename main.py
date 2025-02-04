import torch
from torch.utils.data import DataLoader

from datasets.SlideSeperatedCSVDataset import SlideSeperatedCSVDataset
from models.mlp import MLPBinaryClassifier
from models.resnet import Resnet18BinaryClassifier
from train import train_classifier
from utils import reduce_dataset, split_dataset, undersample_dataset

slides_root_dir = "data/whole-slides/gut"
annotations_root_dir = "data/annotations/json"
candidates_dataset_dir = "output/candidates"
model_output_dir = "output/models"
PretrainedModelClass = Resnet18BinaryClassifier
features_csv_file_name = f"{PretrainedModelClass.get_pretrained_model_name()}_{PretrainedModelClass.pretrained_output_size}_features.csv"
print(f"{PretrainedModelClass.get_pretrained_model_name()}: {PretrainedModelClass.pretrained_output_size} features")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
batch_size = 4096
dataset = SlideSeperatedCSVDataset(candidates_dataset_dir, features_csv_file_name)
dataset = reduce_dataset(dataset, discard_ratio=0.0)
train_dataset, validation_dataset = split_dataset(dataset, train_ratio=0.9)
train_dataset = undersample_dataset(train_dataset)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
validation_loader = DataLoader(validation_dataset,
                               batch_size=batch_size,
                               shuffle=True)

model = MLPBinaryClassifier(in_features=PretrainedModelClass.pretrained_output_size, hidden_layers=1, units_per_layer=PretrainedModelClass.pretrained_output_size,
                            dropout=0.3, focal_alpha=0.9, focal_gamma=3.0)

print(f"Dataset: {len(train_dataset):,} training, {len(validation_dataset):,} validation")

model = model.to(device)
model, model_metrics = train_classifier(model, train_loader, validation_loader, device,
                                        start_learning_rate=0.000075,
                                        max_epochs=20,
                                        checkpoint_every=1,
                                        eval_every=1)
