#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC

clf = SVC(kernel='rbf', C=10000)

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print("training time: " + str(round(time()-t0, 3)) + "s")

t1 = time()
pred = clf.predict(features_test)
print("test time: " + str(round(time()-t1, 3)) + "s")

for x in [10, 26, 50]:
    if pred[x] == 0:
        person = "Sara"
    elif pred[x] == 1:
        person = "Chris"
    # person = "Sara" if pred[x] == 0 else "Chris"
    print(person, "wrote the", x, "th Email.")

print("Predictions: ", pred[10], pred[26], pred[50])
no_of_chris = sum(pred)
print("Chris wrote", no_of_chris, "Emails.")
print("Sara wrote", len(pred)-no_of_chris, "Emails.")
print("Predictions for Chris: ", len([i for i in pred if i == 1]))

print("Accuracy: ", clf.score(features_test, labels_test))
#########################################################


