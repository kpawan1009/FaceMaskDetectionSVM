import cv2
import numpy as np
import pickle

haar_data=cv2.CascadeClassifier('data.xml')

with_mask = np.load('with_mask.npy')
without_mask=np.load('without_mask.npy')

names={0:'Mask',1:'No Mask'}
with_mask=with_mask.reshape(200,150*150*3)
without_mask=without_mask.reshape(200,150*150*3)

X = np.r_[with_mask,without_mask]

labels = np.zeros(X.shape[0])

labels[200:] = 1.0

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=.7,stratify=labels,random_state=5)

svm=SVC()
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)


print(accuracy_score(y_test,y_pred))

# 100% accuracy means overfitting
#
haar_data=cv2.CascadeClassifier('data.xml')


pickle.dump(svm,open("mymodel","wb"))
