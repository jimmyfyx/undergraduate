import cv2 as cv


'''
Resize an image
'''
img = cv.imread('Photos/alma.jpg')

# the scaling factor to resize the image
scale = 0.6    
# calculate new width and new height
new_width = int(img.shape[1] * scale)
new_height = int(img.shape[0] * scale)
# resize the image
img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)

cv.imshow('alma', img)
cv.waitKey(0)  # display the image until a key is pressed


'''
Resize every frame in a video
'''
# read frames from computer camera (take '0' as argument)
cap = cv.VideoCapture(0)

# show the video frame by frame
while True:
    isTrue, frame = cap.read()

    # the scaling factor to resize the image
    scale = 0.6    
    # calculate new width and new height
    new_width = int(frame.shape[1] * scale)
    new_height = int(frame.shape[0] * scale)
    # resize the image
    frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)
    
    cv.imshow('frame', frame)
    # close the window when the key 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# release the capture
cap.release()
cv.destroyAllWindows()