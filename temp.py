import numpy as np
from PIL import Image

# Set grid and tile sizes
tile_width, tile_height = 32, 32  # example tile size
image_width, image_height = image.size
rows, cols = image_height // tile_height, image_width // tile_width

# Convert the image and mask to NumPy arrays
image_array = np.array(image)
mask_array = np.array(mask)

# Prepare a list to hold cropped tiles
active_tiles = []

# Loop over the grid of tiles
for row in range(rows):
    for col in range(cols):
        # Define tile boundaries
        y1, y2 = row * tile_height, (row + 1) * tile_height
        x1, x2 = col * tile_width, (col + 1) * tile_width

        # Check if this tile overlaps with active regions in the mask
        tile_mask = mask_array[y1:y2, x1:x2]
        if np.any(tile_mask):  # Active region detected
            # Crop and store the corresponding tile from the image
            tile = image_array[y1:y2, x1:x2]
            active_tiles.append(tile)

# Now, active_tiles contains all cropped image tiles that correspond to the mask's active regions
