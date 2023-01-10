'''
Conversion among color spaces
'''
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Photos/alma.jpg')
cv.imshow('alma', img)

'''
** Important **
When a RGB image is read by OpenCV, it is automatically converted to BGR color space. Thus, if we use matplotlib to
display the imported image img as below, we will see color inversion. 
'''
# plt.imshow(img)
# plt.show()

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

'''
Gray scale image cannot be directly converted to HSV. It must first go through Grayscale to BGR first
'''
# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

# HSV to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB --> BGR', lab_bgr)

cv.waitKey(0)