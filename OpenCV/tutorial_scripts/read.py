import cv2 as cv

'''
Read and show an image
'''
img = cv.imread('Photos/alma.jpg')

cv.imshow('alma', img)
cv.waitKey(0)  # display the image until a key is pressed


'''
Read and show a video
'''
# read frames from computer camera (take '0' as argument)
cap = cv.VideoCapture(0)

# show the video frame by frame
while True:
    isTrue, frame = cap.read()
    cv.imshow('frame', frame)
    # close the window when the key 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# release the capture
cap.release()
cv.destroyAllWindows()