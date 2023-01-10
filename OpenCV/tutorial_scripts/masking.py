import cv2 as cv
import numpy as np

img = cv.imread('Photos/alma.jpg')
cv.imshow('alma', img)

# Create a circle-like mask
blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, thickness=-1)

# Create a masked image
masked = cv.bitwise_and(img, img, mask=mask)  # the masked image only displays the region where the mask exist
cv.imshow('Masked', masked)

cv.waitKey(0)
