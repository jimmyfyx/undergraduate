import cv2 as cv
import numpy as np

# Displayed image size according to the resolution of computer's webcam
# Change the image size for different cameras
img_width = 1280
img_height = 720

# read frames from webcam 
cap = cv.VideoCapture(0)
cap.set(3, img_width)
cap.set(4, img_height)


def preprocess_frame(img):
    '''
    Input: img - frame (original image)
    Output: img_canny - canny edges image

    Prepare the canny edges before finding contours
    '''
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), cv.BORDER_DEFAULT)
    img_canny = cv.Canny(img_gray, 200, 200)
    return img_canny


def get_contours(img):
    '''
    Input: img - canny edges image
    Output: biggest_contour - the contour with 4 vertices (the file)

    1. Detect contours
    2. Find the contour with the greatest area and with exactly 4 vertices
    3. Approximate the contour shape using approxPolyDP to make the borders smoothier
    4. Draw the contour with the biggest area on the original image
    '''
    contours, hierarchies = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    biggest_contour = np.array([])
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 5000:
            contour_peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * contour_peri, True)
            if area > max_area and len(approx) == 4:
                max_area = area
                biggest_contour = approx
   
    cv.drawContours(img_copy, biggest_contour, -1, (0, 0, 255), thickness=5)
    return biggest_contour


def reorder_points(biggest_contour):
    '''
    Input: biggest_contour - the detected contour of the file

    To have the 'getPerspectiveTransform' function to work, the order of the points in 'points1' needs to be the same order as 'points2'. 
    This function reorders points in 'biggest_contour'.
    '''
    biggest_contour_reordered = np.zeros((4, 2))
    biggest_contour = np.reshape(biggest_contour, (4, 2))
    # get the upper left and bottom right points
    biggest_contour_reordered[0] = biggest_contour[np.argmin(biggest_contour.sum(axis=1))]
    biggest_contour_reordered[3] = biggest_contour[np.argmax(biggest_contour.sum(axis=1))]
    # get the upper right and bottom left points
    biggest_contour_reordered[1] = biggest_contour[np.argmin(np.diff(biggest_contour, axis=1))]
    biggest_contour_reordered[2] = biggest_contour[np.argmax(np.diff(biggest_contour, axis=1))]

    return biggest_contour_reordered    


def warp_perspective(img, biggest_contour):
    '''
    Inputs:
        > img - frame (original image) 
        > biggest - the detected contour of the file
    Output: result - the resulting warp perspective image of the file

    Use the detected contour information to form a warp view of the file
    '''
    reordered_contour = reorder_points(biggest_contour)
    points1 = np.float32(reordered_contour)
    points2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    matrix = cv.getPerspectiveTransform(points1, points2)
    result = cv.warpPerspective(img, matrix, (1280, 720))

    return result


# display the webcam
while True:
    # read and resize each frame
    isTrue, frame = cap.read()
    frame = cv.resize(frame, (img_width, img_height))
    img_copy = frame.copy()

    # detect contours and get warp view
    img_canny = preprocess_frame(frame)
    biggest_contour = get_contours(img_canny)
    cv.imshow('Original', frame)
    cv.imshow('Canny', img_canny)
    cv.imshow('Detected', img_copy)
    if len(biggest_contour) != 0:
        result = warp_perspective(frame, biggest_contour)
        cv.imshow('Scanned', result)

    # close the window when the key 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# release the capture
cap.release()
cv.destroyAllWindows()