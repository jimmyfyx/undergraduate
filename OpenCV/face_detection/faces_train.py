'''
For the training set, we need input and label
- The input will be the images of each person's face
- The label is self-defined
'''
import os
import cv2 as cv
import numpy as np

# get all people's name in a list (folder names)
# the label of each person is the corresponding index in the 'people' list
people = []
for i in os.listdir(r'..\Faces\train'):    # the 'r' prefix is neccessary here because we want the backlashes only be considered as backlashes
    people.append(i)

print(people)

# create the haar cascade object
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []
def create_train_set():
    for person in people:
        path = os.path.join(r'D:\OpenCV_tut\Faces\train', person)
        label = people.index(person)

        # for each person, read all of the images
        for img_name in os.listdir(path):
            # for each image, capture the face and store in 'features'; also store 'label' in 'labels'
            img_path = os.path.join(path, img_name)
            img = cv.imread(img_path)

            if img is None:
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                features.append(face)
                labels.append(label)

create_train_set()

# instantiate the built-in face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the face_recognizer
features = np.array(features, dtype='object')
labels = np.array(labels)
face_recognizer.train(features, labels)

# save the recognizer
face_recognizer.save('face_trained.yml')
# the npy file is designed to store numpy array on disk
np.save('features.npy', features)  
np.save('labels.npy', labels)

