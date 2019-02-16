#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu

# Import SVC
from sklearn.svm import SVC
import pandas as pd
# Create a support vector classifier
clf = SVC()

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')['label']
y_test = pd.read_csv('y_test.csv')['label']


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# # Fit the classifier using the training data
clf.fit(X_train, y_train)
#
# # Predict the labels of the test set
y_pred = clf.predict(X_test)

# Count the number of correct predictions
n_correct = 0
# print(y_pred[1])

for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        n_correct += 1
print(n_correct)
