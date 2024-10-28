import cv2
import numpy as np

cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


def preprocess(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def main():
    raw_image = cv2.imread('data/segments/big01.jpg')
    image = preprocess(raw_image)
    raw_template = cv2.imread('data/templates/template_1.png', cv2.IMREAD_UNCHANGED)
    template_mask = np.where(raw_template[:, :, 3] > 0, 255, 0).astype(np.uint8)
    template = preprocess(raw_template[:, :, :3])

    # Convert the images to grayscale for better matching accuracy
    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Get the width and height of the template image
    template_w, template_h = template_gray.shape[::-1]

    # Perform template matching using the cv2.matchTemplate function
    # TM_CCOEFF_NORMED is one of the available methods, which normalizes the result
    result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=template_mask)

    # Set a threshold for matching (values close to 1 are more similar)
    threshold = 0.5
    locations = np.where(result >= threshold)

    # Draw rectangles around the matched areas
    for point in zip(*locations[::-1]):
        top_left = point
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
