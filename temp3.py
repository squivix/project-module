import cv2
import numpy as np
import os

cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# Load the binary mask (ensure it is in grayscale mode)
mask = cv2.imread("sliver_temp.png", cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread("orig_temp.png", cv2.IMREAD_COLOR)
annotated_image = cv2.imread("orig_temp.png", cv2.IMREAD_COLOR)

# Define the desired size of each cell
cell_size = 256

# Define a threshold for the minimum number of white pixels to consider
white_pixel_threshold = 100  # Adjust this value based on your needs

# Get the dimensions of the mask
height, width = mask.shape

# Output directory to save cropped cells
output_dir = "output/temp"
os.makedirs(output_dir, exist_ok=True)

# Iterate over the binary mask and crop it into non-overlapping cells
for y in range(0, height, cell_size):
    for x in range(0, width, cell_size):
        # Define the region of interest (ROI)
        roi = mask[y:y + cell_size, x:x + cell_size]

        # Check if the ROI has the same size as the desired cell size
        if roi.shape[0] == cell_size and roi.shape[1] == cell_size:
            # Count the number of white pixels in the ROI
            white_pixel_count = cv2.countNonZero(roi)

            # Save the cropped cell only if it has more white pixels than the threshold
            if white_pixel_count > white_pixel_threshold:
                cropped_filename = f"{output_dir}/cell_{y}_{x}.png"
                cv2.imwrite(cropped_filename, original_image[y:y + cell_size, x:x + cell_size])
                top_left = (x, y)
                bottom_right = (x + cell_size, y + cell_size)
                cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)

cv2.imwrite("annotated_temp.png", annotated_image)
# cv2.imshow("image", annotated_image)
# cv2.waitKey()
# print("Cropping completed!")
