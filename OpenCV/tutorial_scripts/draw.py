import cv2 as cv
import numpy as np

# create a 500 * 500 * 3 black image
blank = np.zeros((500, 500, 3), dtype='uint8')

# 1. Paint the image with a certain color
blank[:] = 0, 255, 0     # paint the whole image by changing the G channel to maximum value (255)
cv.imshow('Green', blank)

blank[200:300, 300:400] = 0, 255, 0    # paint a part of the image to green (a square)
cv.imshow('Green', blank)


# 2. Draw a rectangle
cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=2)    # standard rectangle
cv.imshow('Rectangle', blank)

cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=-1)    # filled rectangle
cv.imshow('Rectangle', blank)


# 3. Draw a circle
cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness=3)   # standard circle
cv.imshow('Circle', blank)


# 4. Draw a line
cv.line(blank, (0, 0), (250, 250), (255, 255, 255), thickness=2)
cv.imshow('Line', blank)


# 5. Write text
cv.putText(blank, 'Hello World', (225, 225), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), thickness=2)
cv.imshow('Text', blank)

cv.waitKey(0)

