import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip = 0
face_data = []
dataset_path = './FaceRecognition_Data/'

while True:
    retval, frame = cap.read()
    if retval is False:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 200), 2)

        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))
        skip += 1

    cv2.imshow('Frame', frame)
    cv2.imshow('Face Section', face_section)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file_name = input('Enter Your Name\n')

face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
if os.path.isdir(dataset_path):
    np.save(dataset_path + file_name + '.npy', face_data)
    print('Data successfully saved at ' + dataset_path + file_name + '.npy')
else:
    print('Directory Does Not Exist\nMaking Directory')
    os.mkdir(dataset_path)
    np.save(dataset_path + file_name + '.npy', face_data)
    print('Data successfully saved at ' + dataset_path + file_name + '.npy')

cap.release()
cv2.destroyAllWindows()
