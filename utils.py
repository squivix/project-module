import math
import random

import torch
from matplotlib import pyplot as plt
from torchvision.transforms import v2

from models.cnn import CnnModel


def plot_model_metrics(model_metrics):
    plt.plot(model_metrics["train_losses"], label="train loss")
    plt.plot(model_metrics["test_losses"], label="test loss")
    plt.plot(model_metrics["train_accuracies"], label="train accuracy")
    plt.plot(model_metrics["test_accuracies"], label="test accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    plt.title('Loss & Accuracy in training and testing by epoch')
    plt.xlabel('Epoch')


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


def rescale_data_transform(old_min, old_max, new_min, new_max, should_round=False):
    old_range = old_max - old_min
    new_range = new_max - new_min

    def rescale_lambda(old_val):
        new_val = ((old_val - old_min) * new_range) / old_range + new_min
        if should_round:
            new_val = torch.round(new_val)
        return new_val

    return v2.Lambda(rescale_lambda)


def apply_model(model: CnnModel, test_dataset, indexes, device):
    examples = torch.stack([test_dataset[idx] for idx in indexes]).to(device)
    with torch.no_grad():
        test_logits = model.forward(examples)
        predicted_labels = model.predict(test_logits)

    plot_grid_size = int(math.ceil(math.sqrt(len(examples))))
    text_offset = 0.075
    fig, axes = plt.subplots(plot_grid_size, plot_grid_size, figsize=(10, 10))
    axes = axes.flatten()
    transform = v2.Lambda(lambda t: (t / 255))
    for i, example_index in enumerate(indexes):
        image = transform(test_dataset.get_raw_image(example_index))
        axes[i].imshow(image.permute(1, 2, 0).numpy(force=True), cmap='gray')
        axes[i].axis('off')  # Hide axes
        predicted_label_indexes = predicted_labels[i].nonzero(as_tuple=True)[0].tolist()

        axes[i].annotate(f"{example_index} {test_dataset.image_file_names[example_index]}", (0.5, -0.1), xycoords='axes fraction',
                         ha='center', va='top', fontsize=10, color='black')
        for j, label_idx in enumerate(predicted_label_indexes):
            axes[i].annotate(test_dataset.classes[label_idx], (0.5, -0.2 - (j * text_offset)), xycoords='axes fraction',
                             ha='center', va='top', fontsize=10,
                             color='red')

    for i in range(len(examples), plot_grid_size ** 2):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


