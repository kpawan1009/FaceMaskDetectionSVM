import cv2
import numpy as np

haar_data=cv2.CascadeClassifier('data.xml')

capture = cv2.VideoCapture(0)
data = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (15, 112, 1000), 4)
            face = img[y:y + h, x:x + w, :]
            face = cv2.resize(face, (150, 150))
            print(len(data))
            if len(data) < 200:
                data.append(face)

        cv2.imshow('Result Image', img)
        if cv2.waitKey(1) == 27 or len(data) >= 200:
            break
capture.release()
cv2.destroyAllWindows()

np.save('without_mask.npy',data)

