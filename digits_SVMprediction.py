# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 20:38:18 2017

@author: dgarg
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 22:10:53 2016

@author: dgarg
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.svm import SVC

import csv


mydata = np.genfromtxt(r'C:\DGLearning\Kaggle\Digits\train.csv', delimiter = ',')
y = mydata[1:,0]
X = mydata[1:,1:]

#split data into training and validation sets
Xtrain,  Xtest, ytrain, ytest = train_test_split(X, y, random_state = 0)

#testing data
mytestdata = np.genfromtxt(r'C:\DGLearning\Kaggle\Digits\test.csv', delimiter = ',')

#Use Support vector machines with a polynomial kernel.
clf = SVC(kernel='poly')
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
plt.imshow(metrics.confusion_matrix(ypred, ytest), interpolation='nearest', cmap=plt.cm.binary)
f1_score = metrics.f1_score(ytest, ypred, average='weighted')

print 'F1 score with SVC polynomial kernels:  %s' %f1_score
ytestpred = clf.predict(mytestdata[1:, 0:])

    #open file for output
with open(r'C:\DGLearning\Kaggle\Digits\SVM_prediction.csv', 'wb') as outfile:
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(["ImageID", "Label"])
    imageID = 1
    for label in ytestpred:
        csv_writer.writerow([imageID, label])
        imageID = imageID+1      