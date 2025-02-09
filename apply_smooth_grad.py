import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Saliency, NoiseTunnel
from torchvision.io import ImageReadMode, read_image

from transforms.default_image_transform import default_image_transform


def smoothgrad_threshold_contours(model, input_image, target_class, device):
    model.to(device)
    input_image = input_image.to(device)
    input_image.requires_grad = True

    saliency = Saliency(model)
    smoothgrad = NoiseTunnel(saliency)
    attributions = smoothgrad.attribute(input_image, nt_samples=50, nt_type='smoothgrad', target=None)

    # Convert attributions to positive values using ReLU
    attributions = torch.clamp(attributions, min=0).cpu().detach().numpy()
    # Average across channels if input has multiple channels
    attributions = np.mean(attributions, axis=1)[0]  # Assuming single batch
    # Compute the threshold value at 99.8% percentile
    threshold = np.percentile(attributions, 99.8)
    # Create a mask by thresholding the attributions
    mask = attributions > threshold

    # Apply the mask to the input image (assume input is normalized)
    input_image_np = input_image.cpu().detach().numpy()[0].transpose(1, 2, 0)
    input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())  # Normalize to [0, 1]
    # masked_image = input_image_np * mask[..., np.newaxis]
    # Dilate the gradient mask to connect components
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = (input_image_np * 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)  # Green contours

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(input_image_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("SmoothGrad Mask")
    plt.imshow(attributions, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Contours Highlighted")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = torch.load("model.pickle").to(device)
model.eval()

# Load and preprocess the image
img_path = 'data/candidates/positive/522934_65132_26460_256_256.png'

original_image = read_image(img_path, mode=ImageReadMode.RGB)
original_image = original_image.float() / 255.0

input_tensor = default_image_transform(original_image).unsqueeze(0).to(device)
smoothgrad_threshold_contours(input_image=input_tensor, model=model, target_class=1, device=device)
