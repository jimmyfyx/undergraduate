'''
Split and Merge color channels
'''
import cv2 as cv
import numpy as np

img = cv.imread('Photos/alma.jpg')
cv.imshow('alma', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

# Split color channels
# b, g, r are all gray scale, representing the intensity of blue, green, or red of the original image
b,g,r = cv.split(img)
cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('Merged Image', merged)

cv.waitKey(0)