import math
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2
from tqdm import tqdm


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


def mlp_apply(model, test_indexes, test_dataset):
    examples = test_dataset.data[test_indexes]
    true_labels = test_dataset.targets[test_indexes]
    with torch.no_grad():
        test_logits = model.forward(examples)
        predicted_labels = torch.max(torch.softmax(test_logits, 1), dim=1)[1]
        correct_count = torch.sum((predicted_labels == true_labels).long())
        print(f"Accuracy on the {len(examples)} examples: {correct_count}/{len(examples)}")

    plot_grid_size = int(math.ceil(math.sqrt(len(examples))))
    fig, axes = plt.subplots(plot_grid_size, plot_grid_size, figsize=(10, 10))
    axes = axes.flatten()
    for i, image in enumerate(examples):
        axes[i].imshow(image.numpy(force=True), cmap='gray')
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
        test_subset = Subset(dataset, test_indices)
        test_subset.labels = test_labels
        return train_subset, test_subset
    else:
        return dataset, Subset(dataset, [])


def undersample_dataset(dataset: Dataset, target_size: int = None):
    labels = dataset.labels
    label_indices = defaultdict(list)

    # Group indices by class
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)

    if target_size is None:
        target_size = min(len(indices) for indices in label_indices.values())

    undersampled_indices = []
    for indices in label_indices.values():
        undersampled_indices.extend(np.random.choice(indices, target_size, replace=False).tolist())
    subset = Subset(dataset, undersampled_indices)
    subset.labels = dataset.labels[undersampled_indices]
    return subset
