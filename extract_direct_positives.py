import json
import math
import os
from pathlib import Path

import cv2

from utils import absolute_bbox_to_relative

if hasattr(os, 'add_dll_directory'):
    # Windows
    OPENSLIDE_PATH = os.path.join(os.path.abspath(os.getcwd()),
                                  "libs/openslide-bin-4.0.0.3-windows-x64/bin")
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import numpy as np
import matplotlib.pyplot as plt


def minimum_enclosing_rectangle(polygon_points, cell_size=256):
    # Step 1: Calculate the centroid of the polygon
    n = len(polygon_points)
    x_centroid = sum(x for x, y in polygon_points) / n
    y_centroid = sum(y for x, y in polygon_points) / n

    # Step 2: Determine the bounding box of the polygon
    min_x = min(x for x, y in polygon_points)
    max_x = max(x for x, y in polygon_points)
    min_y = min(y for x, y in polygon_points)
    max_y = max(y for x, y in polygon_points)

    # Step 3: Calculate width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Step 4: Determine the side length of the enclosing square (multiple of l)
    max_side = max(width, height)
    side_length = math.ceil(max_side / cell_size) * cell_size

    # Step 5: Expand the rectangle to be centered on the centroid
    half_side = side_length / 2
    rect_min_x = int(x_centroid - half_side)
    rect_min_y = int(y_centroid - half_side)

    # Step 6: Return rectangle in (x_min, y_min, width, height) format
    return (rect_min_x, rect_min_y, side_length, side_length)


def adjust_and_split_crop(annotation, slide, cell_size=256, display=False):
    min_x, min_y, width, height = annotation["bbox"]
    # new_min_x, new_min_y, new_width, new_height = minimum_enclosing_rectangle(annotation["points"], cell_size=cell_size)
    min_x, min_y, width, height = bbox
    # Calculate the required padding
    pad_width = (cell_size - (width % cell_size)) % cell_size
    pad_height = (cell_size - (height % cell_size)) % cell_size

    # Distribute padding symmetrically
    pad_x1 = pad_width // 2
    pad_y1 = pad_height // 2

    # Adjust crop coordinates
    new_min_x = min_x - pad_x1
    new_min_y = min_y - pad_y1
    new_width = width + pad_width
    new_height = height + pad_height

    print(f"{Path(slide._filename).stem} {(min_x, min_y, width, height)} -> {(new_min_x, new_min_y, new_width, new_height)}")
    # Split into tiles
    tiles = []
    tiles_x = new_width // cell_size
    tiles_y = new_height // cell_size
    roi = np.array(slide.read_region((new_min_x, new_min_y), 0, (new_width, new_height)))

    for i in range(tiles_x):
        for j in range(tiles_y):
            tile_x = new_min_x + i * cell_size
            tile_y = new_min_y + j * cell_size
            tiles.append((tile_x, tile_y, cell_size, cell_size))

            # Draw rectangles on the ROI
            plt.plot([i * cell_size, i * cell_size + cell_size], [j * cell_size, j * cell_size], 'k-', lw=1)  # Top edge
            plt.plot([i * cell_size, i * cell_size + cell_size], [j * cell_size + cell_size, j * cell_size + cell_size], 'k-', lw=1)  # Bottom edge
            plt.plot([i * cell_size, i * cell_size], [j * cell_size, j * cell_size + cell_size], 'k-', lw=1)  # Left edge
            plt.plot([i * cell_size + cell_size, i * cell_size + cell_size], [j * cell_size, j * cell_size + cell_size], 'k-', lw=1)  # Right edge


    if display:
        vis_x, vis_y, vis_w, vis_h = absolute_bbox_to_relative((min_x, min_y, width, height), (new_min_x, new_min_y, new_width, new_height))

        # Draw the adjusted bounding box
        plt.plot([vis_x, vis_x + vis_w], [vis_y, vis_y], 'g-', lw=2)  # Top edge
        plt.plot([vis_x, vis_x + vis_w], [vis_y + vis_h, vis_y + vis_h], 'g-', lw=2)  # Bottom edge
        plt.plot([vis_x, vis_x], [vis_y, vis_y + vis_h], 'g-', lw=2)  # Left edge
        plt.plot([vis_x + vis_w, vis_x + vis_w], [vis_y, vis_y + vis_h], 'g-', lw=2)  # Right edge

        plt.imshow(roi)
        plt.show()

    return tiles


annotations_dir = "data/annotations/json"
slide_name_to_annotations = {}
for annotations_file_name in os.listdir(annotations_dir):
    with open(f"{annotations_dir}/{annotations_file_name}") as f:
        annotations = json.load(f)
        slide_filename = Path(annotations_file_name).stem
        slide_name_to_annotations[slide_filename] = annotations
slides_dir = "data/whole-slides/gut"
output_dir = "output/direct-positives"
os.makedirs(output_dir, exist_ok=True)
ws = []
hs = []
for slide_filename, annotations in slide_name_to_annotations.items():
    slide = openslide.OpenSlide(f"{slides_dir}/{slide_filename}.svs")
    for i, annotation in enumerate(annotations):
        bbox = annotation["x_min"], annotation["y_min"], annotation["width"], annotation["height"]
        annotation["bbox"] = bbox
        ws.append(bbox[-2])
        hs.append(bbox[-1])

        crops = adjust_and_split_crop(annotation, slide)
        if len(crops) > 1:
            continue
        # print(slide_filename, bbox)
        for crop in crops:
            x, y, w, h = crop
            roi = np.array(slide.read_region((x, y), 0, (w, h)))
            cv2.imwrite(f"{output_dir}/{slide_filename}_{x}_{y}_{w}_{h}.png", roi)
print(max(ws))
print(max(hs))
list1 = ws
list2 = hs
# Calculate mean and median for each list
mean_list1, median_list1 = np.mean(list1), np.median(list1)
mean_list2, median_list2 = np.mean(list2), np.median(list2)

# Create a figure with two subplots (rows)
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Histogram for List 1
bins = 100
axs[0].hist(list1, bins=bins, alpha=0.7, color='blue')
axs[0].axvline(mean_list1, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_list1:.2f}')
axs[0].axvline(median_list1, color='orange', linestyle='-', linewidth=1.5, label=f'Median: {median_list1:.2f}')
axs[0].set_title('Histogram of List 1')
axs[0].set_ylabel('Frequency')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# Histogram for List 2
axs[1].hist(list2, bins=bins, alpha=0.7, color='green')
axs[1].axvline(mean_list2, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_list2:.2f}')
axs[1].axvline(median_list2, color='orange', linestyle='-', linewidth=1.5, label=f'Median: {median_list2:.2f}')
axs[1].set_title('Histogram of List 2')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.show()
