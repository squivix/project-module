import math

import torch
from matplotlib import pyplot as plt


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
