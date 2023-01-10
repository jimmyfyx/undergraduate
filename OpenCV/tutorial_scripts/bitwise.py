'''
Bitwise Operation on Images
'''
import cv2 as cv
import numpy as np

# Create a 500 * 500 one channel black image
blank = np.zeros((500, 500), dtype='uint8')

# One image with a rectangle and another with a circle
rectangle = cv.rectangle(blank.copy(), (30, 30), (470, 470), 255, thickness=-1)   # since the image only has one channel, we can just use a number to specify color
circle = cv.circle(blank.copy(), (250, 250), 200, 255, thickness=-1)
cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

# bitwise AND --> intersecting regions
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and)

# bitwise OR --> non-intersecting and intersecting regions
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwise_or)

# bitwise XOR --> non-intersecting regions
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise XOR', bitwise_xor)

# bitwise NOT
bitwise_not = cv.bitwise_not(circle)
cv.imshow('Circle NOT', bitwise_not)

'''
White can be considered as 1 and black as 0
'''

cv.waitKey(0)
