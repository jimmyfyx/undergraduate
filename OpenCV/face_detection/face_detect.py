'''
Detect Faces by Cascade Classifier provided by OpenCV

The Cascade Classifier is very sensitive to noises, which means it can falsely recognizes faces
'''
import cv2 as cv

img = cv.imread(r'../Photos/face.jpg')
cv.imshow('Face', img)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Face', gray)

# Use the cascade classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)  # faces is a list, each element is also a list containing the rectangle that captures a face [x, y, w, h]
## Tune the parameters scaleFactor and minNeighbors will weaken the effect of noises

# Draw the rectangle(s) on the original image
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)