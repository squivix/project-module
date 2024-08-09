import math

import torch
from matplotlib import pyplot as plt
from torchvision.transforms import v2


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
    return accuracy, precision, recall, f1


def rescale_data_transform(old_min, old_max, new_min, new_max, should_round=False):
    old_range = old_max - old_min
    new_range = new_max - new_min

    def rescale_lambda(old_val):
        new_val = ((old_val - old_min) * new_range) / old_range + new_min
        if should_round:
            new_val = torch.round(new_val)
        return new_val

    return v2.Lambda(rescale_lambda)
