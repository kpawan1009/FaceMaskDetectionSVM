import cv2
import numpy as np
import pickle


names={0:'Mask',1:'No Mask'}
svm=pickle.load(open("mymodel",'rb'))
haar_data=cv2.CascadeClassifier('data.xml')

capture=cv2.VideoCapture(0)
data=[]
while True:
    flag , img = capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h,x:x+w,:]
            face = cv2.resize(face,(150,150))
            face=face.reshape(1,-1)
            pred=svm.predict(face)[0]
            n=names[int(pred)]
            print(n)
        cv2.imshow('Result Image',img)
        if cv2.waitKey(1)==27 :
            break
capture.release()
cv2.destroyAllWindows()