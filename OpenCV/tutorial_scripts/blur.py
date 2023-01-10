'''
Different Blurring Methods
'''
import cv2 as cv

img = cv.imread('Photos/alma.jpg')
cv.imshow('Alma', img)

# Averaging
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)

# Gaussian Blur
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian Blur', gauss)

# Median Blur
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# Bilateral
bilateral = cv.bilateralFilter(img, 50, 35, 25)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)