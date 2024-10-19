import cv2
import numpy as np

cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

raw_image = cv2.imread('data/segments/big02.jpg')
image = cv2.GaussianBlur(raw_image, (15, 15), 0)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
_, otsu_mask = cv2.threshold(hsv_image[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

dilated = cv2.dilate(otsu_mask, np.ones((7, 7), np.uint8) , iterations=1)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow('image', image)
cv2.waitKey()
# image = cv2.GaussianBlur(raw_image, (7, 7), 0)
# # cv2.imshow("image", image)
#
# lower = np.array([127, 36, 0])
# upper = np.array([175, 255, 140])
# mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lower, upper)
# mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
# mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
# result = cv2.bitwise_and(image, image, mask=mask)
#
# cv2.imshow('image', cv2.hconcat([cv2.bitwise_and(image, image, mask=mask1),
#                                  cv2.bitwise_and(image, image, mask=mask2)]))
# cv2.waitKey()
