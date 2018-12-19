import cv2
import os
import numpy as np


def distance(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum())


def knn(train, test, k=5):
    vals = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        vals.append((d, iy))

    dk = sorted(vals, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]

    new_vals = np.unique(labels, return_counts=True)

    index = np.argmax(new_vals[1])
    pred = new_vals[0][index]

    return pred


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip = 0
face_data = []
label = []
dataset_path = './FaceRecognition_Data/'

class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        print('Loaded ' + fx)
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)


face_dataset = np.concatenate(face_data, axis=0)
face_label = np.concatenate(label, axis=0).reshape((-1, 1))

trainset = np.concatenate((face_dataset, face_label), axis=1)


while True:
    ret, frame = cap.read()
    if ret is False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for face in faces:
        x, y, w, h = face
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))
        out = knn(trainset, face_section.flatten())
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
