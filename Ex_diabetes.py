#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:18:24 2018

@author: suvasama
"""

#------------------------------------------------------------------------------

# EVALUATING A CLASSIFICATION MODEL
# with Pima Indian Diabetes dataset from the UCI Maschine Learning Reposity

#------------------------------------------------------------------------------

# IMPORT PACKAGES

import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import binarize


import matplotlib.pyplot as plt


#------------------------------------------------------------------------------

# READ DATA INTO THE DATA FRAME

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header = None, names = col_names)

# print the first 5 rows of data
print(pima.head())
print("")


#------------------------------------------------------------------------------

# PREDICTING DIABETES STATUS OF A PATIENT GIVEN HEALTH MEASUREMENTS

# Define X and y
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
X = pima[feature_cols]; y = pima.label

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# train a logistic regression model on the training set
logreg = LogisticRegression(); logreg.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)


#------------------------------------------------------------------------------

# PREDICTION ACCURACY

# percentage of correct preditions
print("Accuracy score:")
print(metrics.accuracy_score(y_test, y_pred_class))
print("")

# NULL ACCURACY: PREDICT THE MOST FREQUENT CLASS

# examine the class distribution of the testing set (Panda series method)
y_test.value_counts()

# calculate the percentage of ones and zeroes
y_test.mean();  1 - y_test.mean()

# null accuracy: binary classification problems (0/1)
print('Null accuracy: ', max(y_test.mean(), 1 - y_test.mean()))
print('')

# null accuracy: multi-class classification problems
y_test.value_counts().head(1)/len(y_test)

# COMPARING THE TRUE AND PREDICTED RESPONSES
print('True: ', y_test.values[0:25])
print('Pred: ', y_pred_class[0:25])
print('')


#------------------------------------------------------------------------------

# CONFUSION MATRIX

# IMPORTANT: first argument is true values, second predicted values
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
print('')

# save and slice into four pieces
TP = confusion[1,1]; TN = confusion[0,0]; FP = confusion[0,1]; FN = confusion[1,0]
totT = TP + TN; totF = FP + FN; tot = float(totT + totF)

# METRICS COMPUTED FROM CONFUSION MATRIX
print('Classification accuracy: ', totT/tot); # same as: print(metrics.accuracy_score(y_test, y_pred_class))
print('Classification error: ', totF/tot); # same as: print(1 - metrics.accuracy_score(y_test, y_pred_class))

# sensitivity
print('\nTrue pos. rate: ', TP/ float(TP + FN))
print(metrics.recall_score(y_test, y_pred_class))

# specifity
print('\nNegs that were predicted correctly: ', TN/ float(TN + FP))

print('\nFalse pos. rate: ', FP/ float(TN + FP))

# precision
print('\nPos`s predicted correctly: ', TP/ float(TP + FP), ' or ',
      metrics.precision_score(y_test, y_pred_class))
print('')

#------------------------------------------------------------------------------

# ADJUSTING THE CLASSIFICATION THRESHOLD

# print the first 10 predicted responses
print(logreg.predict(X_test)[0:10])
print('')

# print the first 10 predicted probabilities of class memberships
print(logreg.predict_proba(X_test)[0:10, 1])
print('')

# print the first 10 predicted probabilities for class 1
print(logreg.predict_proba(X_test)[0:10,1])
print('')

y_pred_prob = logreg.predict_proba(X_test)[:,1]


#------------------------------------------------------------------------------

# HISTOGRAM OF PREDICTED PROBABILITIES

plt.hist(y_pred_prob, bins = 8); plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes'); plt.ylabel('Frequency')
plt.show()


#------------------------------------------------------------------------------

# DECREASE THE THRESHOLD IN ORDER TO INCREASE THE SENSITIVITY OF THE CLASSIFIER

# predict diabetes if predicted prob greater than 0.3
y_pred_class = binarize([y_pred_prob], 0.3)[0]

# print the first 10 predicted probabilities
print('\npredicted probabilities:\n', y_pred_prob[0:10])

# print the first 10 predicted classes with the lower threshold
print('\npredicted classes: ', y_pred_class[0:10])

# confusion matrix with default threshold of 0.5
print('\nold confusion matrix (threshold 0.5):\n', confusion)

print('\nnew confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred_class))

print('\nSensitivity has increased: used to be 0.24, now ', 46/float(46 + 16))
print('Specifity has decreased: used to be 0.91, now ', 80/float(70 + 50))


#------------------------------------------------------------------------------

# ROC CURVES AND AREA UNDER THE CURVE

# IMPORTANT: first argument is the true values, second predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr); plt.xlim([0.0,1.0]); plt.ylim([0.0,1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False positive rate (1 - Specify)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)
plt.show()

# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity: ', tpr[thresholds > threshold][-1])
    print('Specificity: ', 1 - fpr[thresholds > threshold][-1])
    print('')

evaluate_threshold(0.5)
evaluate_threshold(0.3)

print(metrics.roc_auc_score(y_test, y_pred_prob))
print('')

# calculate cross-validated AUC
print(cross_val_score(logreg, X, y, cv = 10, scoring = 'roc_auc').mean())
    

