import glob
import math
import os
import os.path
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from shapely import Polygon, box
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import InceptionOutputs
from torchvision.transforms import v2
from tqdm import tqdm

from datasets.OversampledDataset import OversampledDataset
from datasets.SlideSeperatedImageDataset import SlideSeperatedImageDataset


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
        # subset.get_item_untransformed = dataset.get_item_untransformed
    else:
        dataset.dataset = dataset
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
        # train_subset.get_item_untransformed = dataset.get_item_untransformed
        test_subset = Subset(dataset, test_indices)
        # test_subset.get_item_untransformed = dataset.get_item_untransformed
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
        undersampled_indices.extend(np.random.choice(indices, min(target_size, len(indices)), replace=False).tolist())
    subset = Subset(dataset, undersampled_indices)
    subset.labels = dataset.labels[undersampled_indices]
    # subset.get_item_untransformed = dataset.get_item_untransformed
    return subset


default_oversample_transforms = v2.Compose([
    v2.ToImage(),
    # v2.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def oversample_dataset(dataset: Dataset, transforms=None, augment_Size: int = None):
    if augment_Size == 0:
        return dataset

    labels = dataset.labels
    label_indices = defaultdict(list)

    # Group indices by class
    for idx, label in enumerate(labels):
        if isinstance(label, torch.Tensor):
            label = label.item()
        label_indices[label].append(idx)

    if augment_Size is None:
        augment_Size = max(len(indices) for indices in label_indices.values())

    minority_label = min(label_indices, key=lambda k: len(label_indices[k]))

    oversampled_indices = np.random.choice(label_indices[minority_label], augment_Size, replace=True).tolist()
    if transforms is None:
        transforms = default_oversample_transforms
    return OversampledDataset(dataset, oversampled_indices, minority_label, transforms)


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


def downscale_points(points, downscale_factor):
    downscale_factor = int(downscale_factor)
    new_points = []
    for point in points:
        new_point = tuple(int(c / downscale_factor) for c in point)
        new_points.append(new_point)

    return new_points


def upscale_bbox(bbox, downscale_factor):
    xmin, ymin, width, height = bbox
    downscale_factor = int(downscale_factor)
    # Downscale each value
    new_xmin = int(xmin * downscale_factor)
    new_ymin = int(ymin * downscale_factor)
    new_width = int(width * downscale_factor)
    new_height = int(height * downscale_factor)

    # Return the new bounding box as a tuple
    return (new_xmin, new_ymin, new_width, new_height)


def is_bbox_1_center_in_bbox_2(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    center_x = x1 + w1 / 2
    center_y = y1 + h1 / 2

    # Check if the center of BBox1 lies within BBox2
    if (x2 <= center_x <= x2 + w2) and (y2 <= center_y <= y2 + h2):
        return True
    else:
        return False


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


def bbox_points_to_wh(bbox):
    (x1, y1), (x2, y2) = bbox
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def bbox_wh_to_points(bbox):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2


def calculate_bbox_overlap(bbox1, bbox2):
    if len(bbox1) == 2 and len(bbox2) == 2:
        bbox1 = bbox_points_to_wh(bbox1)
        bbox2 = bbox_points_to_wh(bbox2)

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_br, y1_br = x1 + w1, y1 + h1
    x2_br, y2_br = x2 + w2, y2 + h2

    x_int_left = max(x1, x2)
    y_int_top = max(y1, y2)
    x_int_right = min(x1_br, x2_br)
    y_int_bottom = min(y1_br, y2_br)

    if x_int_right <= x_int_left or y_int_bottom <= y_int_top:
        return 0.0

    intersect_w = x_int_right - x_int_left
    intersect_h = y_int_bottom - y_int_top

    intersect_area = intersect_w * intersect_h
    bbox1_area = w1 * h1

    return intersect_area / bbox1_area


def relative_bbox_to_absolute(target_bbox, reference_bbox):
    xmin1, ymin1, _, _ = reference_bbox
    xmin2, ymin2, width2, height2 = target_bbox
    xmin2_absolute = xmin1 + xmin2
    ymin2_absolute = ymin1 + ymin2
    return (xmin2_absolute, ymin2_absolute, width2, height2)


def absolute_bbox_to_relative(target_bbox, reference_bbox):
    xmin1, ymin1, w1, h1 = target_bbox
    xmin2, ymin2, _, _ = reference_bbox
    xmin1_in_bbox2 = xmin1 - xmin2
    ymin1_in_bbox2 = ymin1 - ymin2
    return (xmin1_in_bbox2, ymin1_in_bbox2, w1, h1)


def absolute_points_to_relative(target_points, reference_bbox):
    xmin2, ymin2, _, _ = reference_bbox
    new_points = []
    for xmin1, ymin1 in target_points:
        new_points.append((xmin1 - xmin2, ymin1 - ymin2))
    return new_points


def mean_blur_image(image, kernel_size=5):
    if kernel_size is None:
        return image
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def downscale_image(image, factor):
    return cv2.resize(image, (image.shape[0] // factor, image.shape[1] // factor), interpolation=cv2.INTER_AREA)


def crop_cv_image(image, bbox):
    x_min, y_min, width, height = bbox
    return image[y_min:y_min + height, x_min:x_min + width]


def get_polygon_bbox_intersection(points, bbox):
    shape1 = Polygon(points).buffer(0)
    xmin, ymin, width, height = bbox
    xmax, ymax = xmin + width, ymin + height
    bbox_shape = box(xmin, ymin, xmax, ymax)
    intersection = shape1.intersection(bbox_shape)

    shape1_area = shape1.area

    intersection_area = intersection.area

    if shape1_area == 0:
        return 0
    return intersection_area / shape1_area


def sync_data_mislabels():
    file_path = 'data/mislabels/all-mislabels.csv'
    df = pd.read_csv(file_path)
    alt_map = {"positive": "negative", "negative": "positive"}
    for index, row in df.iterrows():
        file_name = f'{"_".join(row["file_name"].split('_')[1:])}_256_256.png'
        file_path = f"data/candidates/{row['classification']}/{file_name}"
        if not os.path.exists(file_path):
            alt_file_path = f"data/candidates/{alt_map[row['classification']]}/{file_name}"
            src_path = alt_file_path
            dst_path = Path(file_path).parent
            print(f"{src_path} -> {dst_path}")
            # shutil.move(src_path,dst_path)


def is_not_mostly_blank(image, non_blank_percentage=0.1, min_saturation=15):
    saturation_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    non_white_pixels = np.sum(saturation_channel > min_saturation)
    return (non_white_pixels / saturation_channel.size) > non_blank_percentage


def is_textured_image(image, min_variance=40.0):
    variance = np.var(image)
    return variance > min_variance


def show_cv2_image(image, title=None):
    if image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.axis('off')
    if title is None:
        title = "image"
    plt.title(title)
    plt.show()


def show_cv2_images(images, titles=None):
    """
    Display multiple OpenCV images in a grid layout using Matplotlib.

    Parameters:
        images (list): List of OpenCV images (NumPy arrays).
        titles (list, optional): List of titles for each image.
    """
    num_images = len(images)

    # Determine the grid size (rows x cols)
    # Calculate rows and columns for a roughly square grid
    cols = math.ceil(math.sqrt(num_images))  # More columns than rows when not a perfect square
    rows = math.ceil(num_images / cols)  # Adjust rows accordingl

    fig, axes = plt.subplots(rows, cols, figsize=(rows * 3, cols * 3), gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=True)
    expected_shape = None
    for idx, ax in enumerate(axes.flat):
        if idx < num_images:
            image = images[idx]

            # Convert image to RGB if necessary
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            expected_shape = image.shape
            ax.imshow(image)

            # Add title if provided
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=10)
        else:
            ax.imshow(np.zeros(expected_shape))
        ax.axis('off')  # Hide empty subplots

    plt.show()


def rotate_image(image, angle):
    """Rotate an image by a specific angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def extract_features_from_dataset(candidates_dataset_dir, pretrained_models, split_by_slide=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    batch_size = 128
    dataset = SlideSeperatedImageDataset(candidates_dataset_dir, with_index=True)
    for ModelClass in pretrained_models:
        pretrained_model = ModelClass.create_pretrained_model()  # ModelClass(hidden_layers=0)
        pretrained_output_size = ModelClass.pretrained_output_size
        pretrained_model_name = ModelClass.get_pretrained_model_name()
        output_csv_filename = f"{pretrained_model_name}_{pretrained_output_size}_features.csv"
        output_csv_path = f"{candidates_dataset_dir}/{output_csv_filename}"

        if os.path.exists(output_csv_path):
            print(f"Found cached {output_csv_path}")
            continue
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        pretrained_model.to(device)
        pretrained_model.eval()  # important

        if split_by_slide:
            slide_folders = []
            for slide_folder in os.listdir(candidates_dataset_dir):
                slide_path = f"{candidates_dataset_dir}/{slide_folder}"
                if os.path.isdir(slide_path):
                    slide_folders.append(slide_path)
                    with open(f"{slide_path}/{output_csv_filename}", mode='w') as f:
                        header = ','.join(["file_path", "slide"] + [f'feature_{i}' for i in range(pretrained_output_size)] + ["label"])
                        f.write(header + '\n')
        else:
            with open(output_csv_path, mode='w') as f:
                header = ','.join(["file_path", "slide"] + [f'feature_{i}' for i in range(pretrained_output_size)] + ["label"])
                f.write(header + '\n')

        # stream-writing each batch to the CSV file
        with torch.no_grad():
            for batch_x, batch_y, idxs in tqdm(dataset_loader, desc=f"Extracting feats from {pretrained_model_name}"):
                batch_x = batch_x.to(device)
                logits = pretrained_model.forward(batch_x)
                if isinstance(logits, InceptionOutputs):
                    logits = logits.logits

                # Move logits to CPU, detach, and convert to numpy
                logits = logits.cpu().detach().numpy()

                # Convert logits to DataFrame and write to CSV in append mode
                batch_df = pd.DataFrame(logits)
                batch_df['label'] = batch_y
                paths = []
                slides = []
                for idx in idxs:
                    file_path = dataset.get_item_file_path(idx)
                    paths.append(file_path)
                    slides.append(Path(file_path).stem.split("_")[0])
                batch_df['file_path'] = paths
                batch_df['slide'] = slides
                cols = batch_df.columns.tolist()
                batch_df = batch_df[cols[-2:] + cols[:-2]]
                if not split_by_slide:
                    with open(output_csv_path, mode='a') as f:
                        batch_df.to_csv(f, header=False, index=False)
                else:
                    for slide in batch_df['slide'].unique():
                        slide_df = batch_df[batch_df['slide'] == slide]
                        with open(f"{candidates_dataset_dir}/{slide}/{output_csv_filename}", mode='a') as f:
                            slide_df.to_csv(f, header=False, index=False)


def clear_features_in_slides(candidates_dataset_dir):
    for slide_folder in os.listdir(candidates_dataset_dir):
        for file in glob.glob(os.path.join(f"{candidates_dataset_dir}/{slide_folder}", "*.csv")):
            os.remove(file)
        for file in glob.glob(os.path.join(f"{candidates_dataset_dir}/{slide_folder}", "*.pickle")):
            os.remove(file)


def bbox_to_points(bbox):
    x_min, y_min, width, height = bbox
    return [(x_min, y_min),
            (x_min + width, y_min),
            (x_min + width, y_min + height),
            (x_min, y_min + height),
            (x_min, y_min)]


def rgb_to_bgr(color):
    return color[2], color[1], color[0]


def filter_points_within_bbox(points, bbox):
    x_min, y_min, width, height = bbox
    x_max, y_max = x_min + width, y_min + height
    return [(x, y) for x, y in points if x_min <= x <= x_max and y_min <= y <= y_max]
