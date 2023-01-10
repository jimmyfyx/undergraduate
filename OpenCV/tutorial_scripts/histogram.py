'''
Plot the pixel intensity distribution of an image (grayscale or color)
'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Photos/alma.jpg')
cv.imshow('alma', img)


'''
Grayscale Histogram
'''
# Convert to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Gray scale histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])

# Gray scale histogram with mask
blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, thickness=-1)
masked = cv.bitwise_and(gray, gray, mask=mask)
cv.imshow('Masked', masked)
gray_hist = cv.calcHist([gray], [0], masked, [256], [0, 256])   # the function needs a masked image, not the mast itself

# Plot the distribution of pixel intensity
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()


'''
Color Histogram
'''
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
colors = ['blue', 'green', 'red']
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()


cv.waitKey(0)

