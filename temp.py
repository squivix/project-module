import cv2
import numpy as np

cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

raw_image = cv2.imread('data/tiles/big01.jpg')

image = cv2.resize(raw_image, (raw_image.shape[0] // 2, raw_image.shape[1] // 2), interpolation=cv2.INTER_AREA)
image = cv2.medianBlur(image, 7, 0)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
binary_mask = cv2.dilate(binary_mask, np.ones((7, 7)), iterations=5)

gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Apply binary thresholding
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask to draw the main contour
mask = np.zeros_like(gray)

# Draw the largest contour on the mask (assuming it's the tissue)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

# Distance transform to get the distance of every pixel to the nearest background pixel
dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

# Normalize for easier visualization and manipulation
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

# Define how thick the sliver should be (in pixels)
sliver_thickness = 100  # Adjust this for desired depth

# Create a binary mask where the distance is within the sliver thickness
_, sliver_mask = cv2.threshold(dist_transform, 0.1, 1.0, cv2.THRESH_BINARY)
_, sliver_mask = cv2.threshold(sliver_mask, 1 - (sliver_thickness / 100), 1.0, cv2.THRESH_BINARY)

# Convert sliver mask to 8-bit for visualization
sliver_mask = (sliver_mask * 255).astype(np.uint8)

# Combine the sliver with the original image for visualization
result = cv2.bitwise_and(raw_image,raw_image, mask=sliver_mask)
cv2.imshow('image', cv2.hconcat([mask, sliver_mask]))
cv2.waitKey()
