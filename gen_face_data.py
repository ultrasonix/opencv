import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("C:/Users/Puneet Jindal 25/ml_course_cb/opencv/haarcascade_frontalface_alt.xml")

dataset_path = "C:/Users/Puneet Jindal 25/ml_course_cb/opencv/face_dataset/"
face_data = []
skip = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    k = 1
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

    skip += 1
    for face in faces[:1]:
        x, y, w, h = face

        offset = 7
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))

        if skip % 10 == 0:
            face_data.append(face_section)

        cv2.imshow(str(k), face_section)
        k += 1

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

person_name = input("Enter your name: ")
np.save(dataset_path + person_name + '.npy', face_data)
print("Data successfully saved at " + dataset_path + person_name + '.npy')


