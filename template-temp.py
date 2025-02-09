import os
from pathlib import Path

import cv2
import numpy as np

from utils import calculate_bbox_overlap, mean_blur_image, downscale_image

cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


def main():  # Define parameters
    # Define parameters
    ds_factor = 2
    template_dir = 'data/templates/gastric-glands/'
    threshold = 0.35
    output_template_size = (200, 200)  # Standardized size for template matching
    n_best_matches = 100000  # Number of top matches to retain
    overlap_threshold = 0.2  # Allowable overlap percentage (20%)

    # Load and downscale the main image
    raw_image = downscale_image(cv2.imread('data/tiles/big02.png'), ds_factor)
    image = mean_blur_image(raw_image)
    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Store all matches with confidence scores
    matches = []

    # Loop through all templates in the directory
    for template_name in os.listdir(template_dir):
        if Path(template_name).suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        template_path = os.path.join(template_dir, template_name)
        print(template_path)

        # Load each template and downscale
        raw_template = downscale_image(cv2.imread(template_path, cv2.IMREAD_UNCHANGED), ds_factor)

        # Resize the template to a fixed size for standardized matching
        if raw_template.shape[2] == 4:  # Check if it has an alpha channel
            template_mask = np.where(cv2.resize(raw_template[:, :, 3], output_template_size) > 0, 255, 0).astype(np.uint8)
            template = mean_blur_image(cv2.resize(raw_template[:, :, :3], output_template_size))
        else:
            template_mask = None
            template = mean_blur_image(cv2.resize(raw_template, output_template_size))

        # Convert the template to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=template_mask)

        # Identify locations where the match exceeds the threshold
        locations = np.where(result >= threshold)

        # Add matches with their confidence scores to the list
        for point, score in zip(zip(*locations[::-1]), result[locations]):
            matches.append((point, score, output_template_size))

    # Sort all matches by score in descending order
    matches = sorted(matches, key=lambda x: x[1], reverse=True)

    # Filter to the top `n_best_matches` and ensure non-overlapping regions
    filtered_matches = []
    for match in matches:
        point, score, (template_w, template_h) = match
        top_left = point
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

        # Check if this region overlaps with any already selected regions
        overlap = False
        for (existing_top_left, existing_bottom_right) in filtered_matches:
            overlap_percentage = calculate_bbox_overlap((top_left, bottom_right), (existing_top_left, existing_bottom_right))
            if overlap_percentage > overlap_threshold:
                overlap = True
                break

        # Add to filtered matches if no overlap is found
        if not overlap:
            filtered_matches.append((top_left, bottom_right))
            # Stop if we have reached the desired number of matches
            if len(filtered_matches) >= n_best_matches:
                break

    # Draw rectangles for each filtered match
    for top_left, bottom_right in filtered_matches:
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Display the final image with non-overlapping, best-matching regions highlighted
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
