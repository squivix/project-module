import os

import cv2
import numpy as np

cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


def preprocess(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def downscale(image, factor):
    return cv2.resize(image, (image.shape[0] // factor, image.shape[1] // factor), interpolation=cv2.INTER_AREA)


def main():  # Define parameters
    ds_factor = 2
    template_dir = 'data/templates/gastric-glands'
    threshold = 0.4

    raw_image = downscale(cv2.imread('data/tiles/big02.png'), ds_factor)
    image = preprocess(raw_image)

    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for template_name in os.listdir(template_dir):
        template_path = os.path.join(template_dir, template_name)
        print(template_path)

        raw_template = downscale(cv2.imread(template_path, cv2.IMREAD_UNCHANGED), ds_factor)

        if raw_template.shape[2] == 4:
            template_mask = np.where(raw_template[:, :, 3] > 0, 255, 0).astype(np.uint8)
            template = preprocess(raw_template[:, :, :3])  # Use only RGB channels
        else:
            template_mask = None
            template = preprocess(raw_template)

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        template_w, template_h = template_gray.shape[::-1]

        result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=template_mask)

        locations = np.where(result >= threshold)

        for point in zip(*locations[::-1]):
            top_left = point
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
