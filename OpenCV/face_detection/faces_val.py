'''
Use the trained face recognizer to predict labels of faces
'''
import cv2 as cv
import numpy as np
import os

people = []
for i in os.listdir(r'..\Faces\train'):    # the 'r' prefix is neccessary here because we want the backlashes only be considered as backlashes
    people.append(i)

# create the cascade classifier object
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# load the trained face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

'''
Comment or uncomment the following two parts to perform face recognition on image or video
'''
## 1. face recognition for a single image ##
# read an image for validation and convert to grayscale
img_val = cv.imread(r'..\Faces\val\ben_afflek\3.jpg')
gray = cv.cvtColor(img_val, cv.COLOR_BGR2GRAY)

# detect faces using haar_cascade
faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# predict labels of faces
for (x, y, w, h) in faces:
    face = gray[y:y + h, x:x + w]
    label, confidence = face_recognizer.predict(face)
    print('The face belongs to {name}, with confidence {confidence}'.format(name=people[label], confidence=confidence))

    cv.putText(img_val, str(people[label]), (30, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img_val, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Face', img_val)

cv.waitKey(0)


## 2. face recognition in video ##
cap = cv.VideoCapture(0)

while True:
    isTrue, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(face)
        print('The face belongs to {name}, with confidence {confidence}'.format(name=people[label], confidence=confidence))

        cv.putText(frame, str(people[label]), (x, y -  10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break


cv.waitKey(0)