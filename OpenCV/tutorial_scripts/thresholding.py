'''
Thresholding of an Image
'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Photos/alma.jpg')
cv.imshow('alma', img)

# Convert to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple thresholding
# thresholding requires the input image to be grayscale
threshold, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)   # if the pixel value is greater than 100, set it to white(255); smaller than 100, set it to black(0)
cv.imshow('Binary', thresh)

# Simple inverse thresholding
threshold, thresh_inv = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)   # if the pixel value is greater than 100, set it to black(0); smaller than 100, set it to white(255)
cv.imshow('Binary Inverse', thresh_inv)

# Adaptive Thresholding
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 9)
cv.imshow('Adaptive Thresholding', adaptive_thresh)


cv.waitKey(0)