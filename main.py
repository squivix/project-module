
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from datasets.CSVDataset import CSVDataset
from models.mlp import MLPBinaryClassifier
from train import train_classifier
from utils import plot_model_metrics
from utils import reduce_dataset, split_dataset, undersample_dataset

device = torch.device('cuda:0' if  torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

batch_size = 256
dataset = CSVDataset("data/features/Resnet18Model_features.csv")
dataset = reduce_dataset(dataset, discard_ratio=0.0)
train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.7)
train_dataset = undersample_dataset(train_dataset)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True, )

model = MLPBinaryClassifier(in_features=512, hidden_layers=1, units_per_layer=2048,
                            dropout=0.2, focal_alpha=0.04, focal_gamma=0.2)

print(f"Dataset: {len(train_dataset):,} training, {len(test_dataset):,} testing")

model = model.to(device)
model, model_metrics = train_classifier(model, train_loader, test_loader, device,
                                        start_learning_rate=0.00001,
                                        min_learning_rate=0.000001,
                                        lr_warmup_steps=25,
                                        max_epochs=100,
                                        checkpoint_every=1,
                                        eval_every=1)
