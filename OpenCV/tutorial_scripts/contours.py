'''
Find Contours of a Image

OpenCV official documents:
https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html#:~:text=What%20are%20contours%3F,better%20accuracy%2C%20use%20binary%20images.
'''
import cv2 as cv
import numpy as np

img = cv.imread('Photos/alma.jpg')
cv.imshow('alma', img)

# convert to gray scale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# OpenCV documents suggest that first apply canny edge detection or thresholding before finding contours
canny = cv.Canny(img_gray, 125, 175)
cv.imshow('Canny Edges', canny)

# Find contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)   # contours is a list; Each element is a list of coordinates

# Draw all the detected contours on a blank image
blank = np.zeros(img.shape, dtype='uint8')
blank = cv.drawContours(blank, contours, -1, (0, 255, 0), thickness=1)  # the thrid argument specify how many contours to draw
cv.imshow('Contours', blank)

cv.waitKey(0)
