import cv2 as cv

img = cv.imread('Photos/alma.jpg')
cv.imshow('Original', img)

# Convert RGB image to grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', img_gray)

# Blur
blur = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascade
edge = cv.Canny(img, 125, 175)
cv.imshow('Edges', edge)

# Dilate image

# Erode image

# Crop image
img_crop = img[100:500, 200:1000, :]    # the first dimension is height, the second is width
cv.imshow('Crop', img_crop)

cv.waitKey(0)

