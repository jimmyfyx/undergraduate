#!/usr/bin/env python

import cv2
import numpy as np

# Params for camera calibration
theta = 0.0
beta = 0.0
tx = 0.0
ty = 0.0


# Function that converts image coord to world coord
def IMG2W(col, row):
    beta = 761.05
    x_c = (row-240)/beta
    y_c = (col-320)/beta
    theta = (3.0128/180)*np.pi

    T_x = 0.311631496   # old 0.321631496                                           
    T_y = 0.01972107 # old 0.01972107

    x_w = np.cos(theta)*(x_c + T_x) + np.sin(theta)*(x_c + T_x)
    y_w = np.cos(theta)*(y_c + T_y) - np.sin(theta)*(y_c + T_y)
    return x_w, y_w


def blob_search(image_raw, color):

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    if(color == 'yellow'):
        # Filter by Color
        params.filterByColor = False

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 8

        # Filter by Circularity
        params.filterByCircularity = False

        # Filter by Inerita
        params.filterByInertia = False

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.7 # For blocks
    else:
        # Filter by Color
        params.filterByColor = False

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 550
        params.maxArea = 800

        # Filter by Circularity
        params.filterByCircularity = False

        # Filter by Inerita
        params.filterByInertia = False

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.7 # For blocks

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Convert the image into the HSV color space
    hsv_image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2HSV)

    if(color == "lavender"):
        lower = (110,20,20)     # lavender lower
        upper = (130,110,170)   # lavender upper
    elif(color == "yellow"):
        lower = (20,120,0)     # yellow lower
        upper = (30,255,255)   # yellow upper
    else:
        lower = (10,100,100)     # orange lower
        upper = (17,255,225)   # orange upper

    # Define a mask using the lower and upper bounds of the target color
    mask_image = cv2.inRange(hsv_image, lower, upper)

    keypoints = detector.detect(mask_image)

    # Find blob centers in the image coordinates
    blob_image_center = []
    num_blobs = len(keypoints)
    for i in range(num_blobs):
        blob_image_center.append((keypoints[i].pt[0],keypoints[i].pt[1]))

    # Draw the keypoints on the detected block
    im_with_keypoints = cv2.drawKeypoints(image_raw, keypoints,0,color=(255,255,255))

    xw_yw = []
    if(num_blobs == 0):
        print("No block found!")
    else:
        # Convert image coordinates to global world coordinate using IM2W() function
        for i in range(num_blobs):
            xw_yw.append(IMG2W(blob_image_center[i][0], blob_image_center[i][1]))

    cv2.namedWindow("Camera View")
    cv2.imshow("Camera View", image_raw)
    cv2.namedWindow("Mask View")
    cv2.imshow("Mask View", mask_image)
    cv2.namedWindow("HSV View")
    cv2.imshow("HSV View", hsv_image)
    cv2.namedWindow("Keypoint View")
    cv2.imshow("Keypoint View", im_with_keypoints)
    cv2.waitKey(2)

    return xw_yw
