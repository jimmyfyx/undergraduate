import cv2 as cv
import numpy as np

img = cv.imread('Photos/alma.jpg')
cv.imshow('alma', img)

'''
Image Translation
'''
def translate(img, x, y):
    trans_mat = np.float32([[1, 0, x], [0, 1, y]])
    output_dim = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, trans_mat, output_dim)

# -x --> left
# -y --> up
# x --> right
# y --> down
trans_img = translate(img, 200, 200)
cv.imshow('Translated', trans_img)


'''
Image Rotation
'''
def rotate(img, angle, rot_point=None):
    (height, width) = img.shape[:2]

    # the rotation point is a 2-dimensional point on the image plane
    if rot_point is None:
        rot_point = (width // 2, height // 2)
    
    rot_mat = cv.getRotationMatrix2D(rot_point, angle, 1.0)
    output_dim = (width, height)
    return cv.warpAffine(img, rot_mat, output_dim)

rot_img = rotate(img, 45)  # positive angles for counterclockwise rotation, negtive for clockwise
cv.imshow('Rotated', rot_img)

##  difference between multiple-times rotation and one-time rotation ##


'''
Flip Image
'''
flip = cv.flip(img, 0)    # 0 for flip w.r.t x-axis; 1 for flip w.r.t. y-axis; -1 for both x and y axis
cv.imshow('Flip', flip)


cv.waitKey(0)