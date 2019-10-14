#!/usr/bin/env python
# coding: utf-8

# In[4]:
# Starting application

#from SVM import SVM   #importing custom SVM 
from sklearn import svm, datasets #importing inbuilt SVM

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30)

m=SVM(kernel='rbf',param=0.1,c=1)
m.fit(X_train,Y_train)

y_predict_train=m.predict(X_train)
y_predict_test=m.predict(X_test)

train_accuracy=accuracy_score(Y_train,y_predict_train)
test_accuracy=accuracy_score(Y_test,y_predict_test)

#using sklearn svm
l=svm.SVC(kernel='rbf',gamma=0.1,C=1)
l.fit(X_train,Y_train)

y_predict_train_o=l.predict(X_train)
y_predict_test_o=l.predict(X_test)
train_accuracy_sklearn=accuracy_score(Y_train,y_predict_train_o)
test_accuracy_sklearn=accuracy_score(Y_test,y_predict_test_o)

print("Using my SVM\ntrain_accuracy: {} \ntest_accuracy: {}".format(train_accuracy, test_accuracy))
print("\n\nUsing Sklearn\ntrain_accuracy: {} \ntest_accuracy: {}".format(train_accuracy_sklearn, test_accuracy_sklearn))


# In[ ]:




