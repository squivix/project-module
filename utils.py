import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2
from tqdm import tqdm

from datasets.OversampledDataset import OversampledDataset


def plot_model_metrics(model_metrics):
    fig, ax = plt.subplots(nrows=2, figsize=(10, 10))

    ax[0].plot(model_metrics["train_epoch"], model_metrics[f"train_loss"], label=f"train loss")
    ax[0].plot(model_metrics["test_epoch"], model_metrics[f"test_loss"], label=f"test loss")
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel('Epoch')
    ax[0].set_title('Loss in training and testing by epoch')

    for metric in ["accuracy", "precision", "recall", "f1"]:
        ax[1].plot(model_metrics["test_epoch"], model_metrics[f"test_{metric}"], label=f"test {metric}")
    ax[1].legend()
    ax[1].grid()
    ax[1].set_title('Confusion metrics in testing by epoch')
    ax[1].set_xlabel('Epoch')
    plt.show()


def apply_model(model, test_dataset, test_indexes, device):
    # examples = test_dataset[test_indexes]
    # true_labels = test_dataset[test_indexes]
    examples, true_labels = next(iter(DataLoader(Subset(test_dataset, test_indexes), batch_size=len(test_indexes))))
    examples = examples.to(device)
    true_labels = true_labels.to(device)
    with torch.no_grad():
        test_logits = model.forward(examples)
        predicted_labels = torch.max(torch.softmax(test_logits, 1), dim=1)[1]
        correct_count = torch.sum((predicted_labels == true_labels).long())
        print(f"Accuracy on the {len(examples)} examples: {correct_count}/{len(examples)}")

        plot_grid_size = int(math.ceil(math.sqrt(len(examples))))
        fig, axes = plt.subplots(plot_grid_size, plot_grid_size, figsize=(10, 10))
        axes = axes.flatten()
        for i, image in enumerate(examples):
            axes[i].imshow(image.permute(1, 2, 0).numpy(force=True), cmap='gray')
            axes[i].axis('off')  # Hide axes
            axes[i].annotate(test_dataset.classes[true_labels[i].item()], (0.5, -0.1), xycoords='axes fraction',
                             ha='center', va='top', fontsize=10,
                             color='green')
            axes[i].annotate(test_dataset.classes[predicted_labels[i].item()], (0.5, -0.2), xycoords='axes fraction',
                             ha='center', va='top', fontsize=10,
                             color='red')

        for i in range(len(examples), plot_grid_size ** 2):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    return


def test_model(model, dataset, device, batch_size=128):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    metrics = ["accuracy", "loss", "precision", "recall", "f1", "mcc"]
    test_metrics = dict()
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_batches = iter(test_loader)

        batch_test_metrics = {m: np.empty(len(test_batches)) for m in metrics}
        for i, (x_test, y_test) in enumerate(tqdm(test_batches, desc=f"Evaluating model")):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            test_logits = model.forward(x_test)
            test_loss = model.loss_function(test_logits, y_test)
            test_preds = model.predict(test_logits)
            test_accuracy, test_precision, test_recall, test_f1, test_mcc = calc_binary_classification_metrics(y_test,
                                                                                                               test_preds)

            batch_test_metrics["loss"][i] = test_loss
            batch_test_metrics["accuracy"][i] = test_accuracy
            batch_test_metrics["precision"][i] = test_precision
            batch_test_metrics["recall"][i] = test_accuracy
            batch_test_metrics["f1"][i] = test_f1
            batch_test_metrics["mcc"][i] = test_mcc
        for m in metrics:
            test_metrics[m] = batch_test_metrics[m].mean()
    return test_metrics


def divide(num, donim):
    if num == 0:
        return 0.0
    return num / donim


def calc_binary_classification_metrics(true_labels, predicted_labels):
    tp = torch.sum((predicted_labels == 1) & (true_labels == 1)).item()
    tn = torch.sum((predicted_labels == 0) & (true_labels == 0)).item()
    fp = torch.sum((predicted_labels == 1) & (true_labels == 0)).item()
    fn = torch.sum((predicted_labels == 0) & (true_labels == 1)).item()

    accuracy = divide(tp + tn, (tp + tn + fp + fn))
    precision = divide(tp, (tp + fp))
    recall = divide(tp, (tp + fn))
    f1 = divide(2 * precision * recall, (precision + recall))
    mcc = divide((tp * tn) - (fp * fn), math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return accuracy, precision, recall, f1, mcc


def rescale_data_transform(old_min, old_max, new_min, new_max, should_round=False):
    old_range = old_max - old_min
    new_range = new_max - new_min

    def rescale_lambda(old_val):
        new_val = ((old_val - old_min) * new_range) / old_range + new_min
        if should_round:
            new_val = torch.round(new_val)
        return new_val

    return v2.Lambda(rescale_lambda)


def reduce_dataset(dataset: Dataset, discard_ratio=0.0):
    if discard_ratio > 0:
        subset_indices, _, subset_labels, _ = train_test_split(np.arange(len(dataset)),
                                                               dataset.labels,
                                                               test_size=discard_ratio,
                                                               stratify=dataset.labels)
        subset = Subset(dataset, subset_indices)
        subset.labels = subset_labels
        subset.get_item_untransformed = dataset.get_item_untransformed
    else:
        subset = dataset
    return subset


def split_dataset(dataset: Dataset, train_ratio=0.7):
    if train_ratio < 1.0:
        train_indices, test_indices, train_labels, test_labels = train_test_split(np.arange(len(dataset)),
                                                                                  dataset.labels,
                                                                                  train_size=train_ratio,
                                                                                  stratify=dataset.labels)
        train_subset = Subset(dataset, train_indices)
        train_subset.labels = train_labels
        train_subset.get_item_untransformed = dataset.get_item_untransformed
        test_subset = Subset(dataset, test_indices)
        test_subset.get_item_untransformed = dataset.get_item_untransformed
        test_subset.labels = test_labels
        return train_subset, test_subset
    else:
        return dataset, Subset(dataset, [])


def undersample_dataset(dataset: Dataset, target_size: int = None):
    labels = dataset.labels
    label_indices = defaultdict(list)

    # Group indices by class
    for idx, label in enumerate(labels):
        if isinstance(label, torch.Tensor):
            label = label.item()
        label_indices[label].append(idx)

    if target_size is None:
        target_size = min(len(indices) for indices in label_indices.values())

    undersampled_indices = []
    for indices in label_indices.values():
        undersampled_indices.extend(np.random.choice(indices, target_size, replace=False).tolist())
    subset = Subset(dataset, undersampled_indices)
    subset.labels = dataset.labels[undersampled_indices]
    return subset


def oversample_dataset(dataset: Dataset, transforms, target_size: int = None):
    labels = dataset.labels
    label_indices = defaultdict(list)

    # Group indices by class
    for idx, label in enumerate(labels):
        if isinstance(label, torch.Tensor):
            label = label.item()
        label_indices[label].append(idx)

    if target_size is None:
        target_size = max(len(indices) for indices in label_indices.values())

    minority_label = min(label_indices, key=lambda k: len(label_indices[k]))

    oversampled_indices = np.random.choice(label_indices[minority_label], target_size, replace=True).tolist()
    return OversampledDataset(dataset, oversampled_indices, transforms)


def clear_dir(dir_path_string):
    dir_path = Path(dir_path_string)
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
    os.makedirs(dir_path_string, exist_ok=True)


def downscale_bbox(bbox, downscale_factor):
    xmin, ymin, width, height = bbox
    downscale_factor = int(downscale_factor)
    # Downscale each value
    new_xmin = xmin // downscale_factor
    new_ymin = ymin // downscale_factor
    new_width = width // downscale_factor
    new_height = height // downscale_factor

    # Return the new bounding box as a tuple
    return (new_xmin, new_ymin, new_width, new_height)


def is_bbox2_within_bbox1(bbox1, bbox2):
    # Unpacking bbox1 and bbox2
    xmin1, ymin1, width1, height1 = bbox1
    xmin2, ymin2, width2, height2 = bbox2

    # Calculate the bottom-right corners of bbox1 and bbox2
    xmax1, ymax1 = xmin1 + width1, ymin1 + height1
    xmax2, ymax2 = xmin2 + width2, ymin2 + height2

    # Check if bbox2 is inside bbox1
    return (xmin1 <= xmin2 <= xmax1 and
            ymin1 <= ymin2 <= ymax1
            # and
            # xmax1 >= xmax2 and
            # ymax1 >= ymax2
            )


def get_relative_bbox2_within_bbox1(bbox1, bbox2):
    # Unpacking bbox1 and bbox2
    xmin1, ymin1, width1, height1 = bbox1
    xmin2, ymin2, width2, height2 = bbox2

    # Calculate the bottom-right corners of bbox1 and bbox2
    xmax1, ymax1 = xmin1 + width1, ymin1 + height1
    xmax2, ymax2 = xmin2 + width2, ymin2 + height2

    # Check if bbox2 is inside bbox1
    if (xmin1 <= xmin2 <= xmax1 and
            ymin1 <= ymin2 <= ymax1 and
            xmax1 >= xmax2 and
            ymax1 >= ymax2):
        # Calculate relative bbox2 coordinates with respect to bbox1
        x_relative = xmin2 - xmin1
        y_relative = ymin2 - ymin1
        relative_bbox = (x_relative, y_relative, width2, height2)
        return relative_bbox
    return None


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    x, y, width, height = bbox
    top_left = (x, y)
    bottom_right = (x + width, y + height)
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image


def draw_sign(image, is_positive, line_length=100, line_thickness=5):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    if is_positive:
        line_color = (0, 0, 255, 255)
    else:
        line_color = (0, 0, 0, 255)
    # Define the center of the image
    center_x, center_y = width // 2, height // 2

    # Draw horizontal line of the "+" sign
    cv2.line(image,
             (center_x - line_length // 2, center_y),
             (center_x + line_length // 2, center_y),
             line_color,
             line_thickness)
    if is_positive:
        # Draw vertical line of the "+" sign
        cv2.line(image,
                 (center_x, center_y - line_length // 2),
                 (center_x, center_y + line_length // 2),
                 line_color,
                 line_thickness)

    return image
