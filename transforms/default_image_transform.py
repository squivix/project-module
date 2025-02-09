import random

import torch
from torchvision.transforms import v2 as T


class OneOf(torch.nn.Module):
    """Applies one of the given transforms with a given probability."""

    def __init__(self, transforms, p=1.0):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            transform = random.choice(self.transforms)
            img = transform(img)
        return img


augmentation_transforms = T.Compose([
    OneOf([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # T.Lambda(lambda img: T.functional.adjust_rgb(img,
        #                                              torch.randint(-20, 20, (1,)).item(),
        #                                              torch.randint(-20, 20, (1,)).item(),
        #                                              torch.randint(-20, 20, (1,)).item())),
        T.RandomEqualize()
    ], p=0.8),

    OneOf([
        T.GaussianBlur(kernel_size=(3, 5)),
        T.RandomAdjustSharpness(sharpness_factor=2, p=1),
        T.RandomAutocontrast(p=1),
    ], p=0.7),

    # OneOf([
    #     T.Lambda(lambda img: img + torch.randn_like(img) * torch.randint(10, 30, (1,)).item()),
    # ], p=0.5),

    OneOf([
        T.RandomAdjustSharpness(sharpness_factor=random.uniform(0.2, 0.5), p=1),
    ], p=0.6),

    OneOf([
        T.RandomPosterize(bits=random.randint(4, 8), p=1),
        T.RandomEqualize(p=1),
    ], p=0.3),
])

default_image_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
