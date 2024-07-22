import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.cnn import CnnModel
from train import mlp_train
from utils import plot_model_metrics


def main():
    data_dir = 'data/DeepHP'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256
        transforms.ToTensor(),  # Convert the image to a tensor
    ])
    dataset_class_0 = ImageFolder(root=f"{data_dir}/Negative", transform=transform)
    dataset_class_1 = ImageFolder(root=f"{data_dir}/Positive", transform=transform,
                                  target_transform=transforms.Lambda(lambda x: x + 1))  # hack
    batch_size = 512

    class_size = 100_000
    dataset = ConcatDataset([
        Subset(dataset_class_0, np.random.choice(len(dataset_class_0), class_size, replace=False)),
        Subset(dataset_class_1, np.random.choice(len(dataset_class_1), class_size, replace=False)),
    ])

    train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
    # num_workers = 4
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True, )
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True, )

    model = CnnModel(output_dim=2)
    print(model)

    model = model.to(device)
    model, model_metrics = mlp_train(model, train_loader, test_loader, device,
                                     learning_rate=0.001,
                                     max_epochs=50)
    print(model_metrics)
    plot_model_metrics(model_metrics)
    torch.save(model.state_dict(), "./model5.bin")


if __name__ == "__main__":
    main()
