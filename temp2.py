import cv2
import numpy as np


def get_tissue_mask(
        thumbnail_im,
        n_thresholding_steps=1, sigma=0., min_size=500):
    from scipy import ndimage
    from skimage.filters import gaussian, threshold_otsu

    if len(thumbnail_im.shape) == 3:
        thumbnail = 255 - cv2.cvtColor(thumbnail_im, cv2.COLOR_BGR2GRAY)
    else:
        thumbnail = thumbnail_im

    for _ in range(n_thresholding_steps):
        if sigma > 0.0:
            thumbnail = gaussian(
                thumbnail, sigma=sigma,
                output=None, mode='nearest', preserve_range=True)
        try:
            thresh = threshold_otsu(thumbnail[thumbnail > 0])
        except ValueError:
            thresh = 0
        thumbnail[thumbnail < thresh] = 0
    mask = 0 + (thumbnail > 0)
    labeled, _ = ndimage.label(mask)

    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
    discard = np.isin(labeled, unique[counts < min_size])
    discard = discard.reshape(labeled.shape)

    labeled[discard] = 0
    mask = labeled == unique[np.argmax(counts)]
    return labeled, mask


def main():
    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    thumbnail_rgb = cv2.imread('data/segments/big04.jpg')
    labeled, mask = get_tissue_mask(thumbnail_rgb, n_thresholding_steps=1, sigma=0.0, min_size=0)
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    outer_distance = 15  # Outer limit for the sliver (pixels)
    inner_distance = 5  # Inner limit to exclude deeper parts (pixels)

    outer_mask = (dist_transform >= inner_distance).astype(np.uint8)
    inner_mask = (dist_transform > outer_distance).astype(np.uint8)
    sliver_mask = outer_mask - inner_mask
    sliver_mask = (sliver_mask * 255).astype(np.uint8)
    result = cv2.bitwise_and(thumbnail_rgb, thumbnail_rgb, mask=sliver_mask)
    cv2.imshow("image", cv2.hconcat([thumbnail_rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(sliver_mask, cv2.COLOR_GRAY2BGR)]))
    cv2.waitKey()
    cv2.imwrite("sliver_temp.png", sliver_mask)
    cv2.imwrite("orig_temp.png", thumbnail_rgb)


main()
