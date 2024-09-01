import glob
import os
from collections import defaultdict
from copyreg import pickle

from sklearn.model_selection import train_test_split

from datasets.CSVDataset import CSVDataset
from datasets.LabeledImageDataset import LabeledImageDataset


#
# class CustomTrainingImageDataset(Dataset):
#     def __init__(self, data_dir, class_limit=None, image_extension="jpeg"):
#         self.transform = v2.Compose([
#             v2.ToImage(),
#             v2.Resize((256, 256)),
#             rescale_data_transform(0, 255, -1, +1)
#         ])
#
#         self.image_file_names = []
#         self.class_labels = os.listdir(data_dir)
#         self.data_dir = data_dir
#         class_sizes = []
#         class_limit = class_limit if class_limit is not None else float("inf")
#         for class_dir in self.class_labels:
#             img_dir = os.path.join(data_dir, class_dir)
#             class_image_paths = sorted([os.path.join(class_dir, os.path.basename(path)) for path in
#                                         glob.glob(f"{img_dir}/*.{image_extension}")])
#             class_sizes.append(len(class_image_paths))
#             class_limit = min(class_limit, len(class_image_paths))
#             self.image_file_names.append(class_image_paths)
#         self.dataset_size = class_limit * len(self.class_labels)
#
#         self.example_lookup = []
#         for class_index in range(len(self.class_labels)):
#             for image_index in np.random.choice(np.arange(0, class_sizes[class_index]), class_limit, replace=False):
#                 self.example_lookup.append((class_index, image_index.item()))
#
#     def __getitem__(self, idx):
#         class_index, img_index = self.example_lookup[idx]
#         img_path = os.path.join(self.data_dir, self.image_file_names[class_index][img_index])
#         img = read_image(img_path)
#
#         x = self.transform(img)
#         y = class_index
#         return x, y
#
#     def __len__(self):
#         return self.dataset_size

def load_image_data(root_dir):
    samples = []
    labels = []
    for class_index, class_name in enumerate(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        for file_path in sorted([path for path in glob.glob(f"{class_dir}/*.jpeg")]):
            samples.append((file_path, class_index))
            labels.append(class_index)
    return samples, labels


def load_csv_data(csv_file_path, skip_header=True):
    samples = []
    labels = []
    with open(csv_file_path, 'r') as csv_file:
        offset = csv_file.tell()
        i = 0
        line = csv_file.readline().strip()
        while line:
            if i == 0 and skip_header:
                line = csv_file.readline().strip()
                i += 1
                continue
            label = int(line.split(",")[-2])
            samples.append((offset, label))
            labels.append(label)
            offset = csv_file.tell()
            line = csv_file.readline().strip()
            i += 1
    return samples, labels


import pickle


def generate_balanced_dataset(discard_ratio=0.0, test_ratio=0.3, undersample=False):
    with open("data/features/InceptionV3_features.pickle", "rb") as temp:
        cached_samples_labels = pickle.load(temp)
    samples = cached_samples_labels["samples"]
    labels = cached_samples_labels["labels"]
    # Keep, discard split
    if discard_ratio > 0:
        samples, _, labels, _ = train_test_split(samples, labels, test_size=discard_ratio, stratify=labels)

    # Train, test split
    train_samples, test_samples, _, _ = train_test_split(samples, labels, test_size=test_ratio, stratify=labels)

    # Undersample training data
    if undersample:
        training_samples_by_class = defaultdict(list)
        for sample in train_samples:
            training_samples_by_class[sample[1]].append(sample)
        min_class_size = min(len(samples) for samples in training_samples_by_class.values())

        undersampled_train_samples = []
        for class_samples in training_samples_by_class.values():
            undersampled_train_samples.extend(class_samples[:min_class_size])
        train_samples = undersampled_train_samples

    csv_path = "data/features/InceptionV3_features.csv"
    training_dataset = CSVDataset(csv_path, train_samples)
    validation_dataset = CSVDataset(csv_path, test_samples)
    return training_dataset, validation_dataset
