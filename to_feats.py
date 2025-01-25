# import pandas as pd
# import torch
# from torch.utils.data import DataLoader
# from torchvision.transforms import v2
#
# from datasets.LabeledImageDataset import LabeledImageDataset
# from models.resnet import Resnet18Model
# from utils import reduce_dataset, undersample_dataset, oversample_dataset, split_dataset
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f"Device: {device}")
#
# batch_size = 128
# original_dataset = LabeledImageDataset("data/candidates")
# original_dataset = reduce_dataset(original_dataset, discard_ratio=0.0)
# train_dataset, test_dataset = split_dataset(original_dataset, train_ratio=0.7)
# train_dataset = oversample_dataset(undersample_dataset(train_dataset, target_size=None),
#                                    augment_Size=0,
#                                    transforms=v2.Compose([
#                                        v2.ToImage(),
#                                        # v2.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
#                                        v2.RandomHorizontalFlip(p=0.5),
#                                        v2.RandomVerticalFlip(p=0.5),
#                                        v2.RandomRotation(degrees=30),
#                                        # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#                                        # v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#                                        v2.ToDtype(torch.float32, scale=True),
#                                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                                    ]))  # undersample_dataset(train_dataset)
# csv_file_paths = "train_dataset.csv", "test_dataset.csv"
# for i, dataset in enumerate([train_dataset, test_dataset]):
#     dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     output_csv_path = csv_file_paths[i]
#
#     model = Resnet18Model(hidden_layers=0, units_per_layer=128)
#     model.to(device)
#
#     # Open the CSV file and write header (if needed)
#     with open(output_csv_path, mode='w') as f:
#         # Assuming logits have a fixed dimensionality, e.g., 512
#         header = ','.join([f'feature_{i}' for i in range(512)] + ["label"])
#         f.write(header + '\n')
#
#     # Stream-writing each batch to the CSV file
#     with torch.no_grad(), open(output_csv_path, mode='a') as f:
#         for batch_x, batch_y in dataset_loader:
#             batch_x = batch_x.to(device)
#             logits = model.pretrained_model.forward(batch_x)
#
#             # Move logits to CPU, detach, and convert to numpy
#             logits = logits.cpu().detach().numpy()
#
#             # Convert logits to DataFrame and write to CSV in append mode
#             batch_df = pd.DataFrame(logits)
#             batch_df['label'] = batch_y
#
#             batch_df.to_csv(f, header=False, index=False)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.LabeledImageDataset import LabeledImageDataset
from models.resnet import Resnet18Model, Resnet50Model
from utils import reduce_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

batch_size = 128
original_dataset = LabeledImageDataset("data/candidates", with_index=True)
dataset = reduce_dataset(original_dataset, discard_ratio=0.0)

dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = Resnet18Model(hidden_layers=0, units_per_layer=128)
model.to(device)

output_csv_path = f"output/{model.__class__.__name__}_features.csv"
# Open the CSV file and write header (if needed)
with open(output_csv_path, mode='w') as f:
    header = ','.join([f'feature_{i}' for i in range(model.pretrained_output_size)] + ["label", "file_path"])
    f.write(header + '\n')

# Stream-writing each batch to the CSV file
file_paths = np.array(dataset.dataset.file_paths)
with torch.no_grad(), open(output_csv_path, mode='a') as f:
    for batch_x, batch_y, idx in tqdm(dataset_loader):
        batch_x = batch_x.to(device)
        logits = model.pretrained_model.forward(batch_x)

        # Move logits to CPU, detach, and convert to numpy
        logits = logits.cpu().detach().numpy()

        # Convert logits to DataFrame and write to CSV in append mode
        batch_df = pd.DataFrame(logits)
        batch_df['label'] = batch_y
        batch_df['file_path'] = file_paths[idx]
        batch_df.to_csv(f, header=False, index=False)
