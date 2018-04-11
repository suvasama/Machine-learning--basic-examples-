#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:14:28 2018

@author: suvasama
"""

#------------------------------------------------------------------------------

# IMPORT PACKAGES

# pandas for handling data
import pandas as pd

# KNN classifier class
from sklearn.neighbors import KNeighborsClassifier

# logistic regression class
from sklearn.linear_model import LogisticRegression

# metrics for classification accuracy
from sklearn import metrics

# train/ test split
from sklearn.cross_validation import train_test_split

# k-fold cross validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

# gridserach for parameter tuning 
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

# scientific plotting library
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------

# IMPORT DATA

# read the iris data into a pandas DataFrame, including column names
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv('iris.data.txt', names=col_names)

# map species to a numeric value
iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# create X (features) three different ways
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# create y (response)
y = iris.species_num

# activate to check the shape of X and y
print(X.shape)     # 150 by 4
print(y.shape)     # 150 (must match first dimension of X)


#------------------------------------------------------------------------------

# KNN ESTIMATOR

# Use one nearest neighborhood

# Initiate the estimator
knn1 = KNeighborsClassifier(n_neighbors = 1)

# Fit the model with data
knn1.fit(X,y)

# Predict the response for a new observation
print(knn1.predict([[3, 5, 4, 2]]))

# You can predict two samples at the same time
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn1.predict(X_new))

# Use five nearest neighborhoods

knn5 = KNeighborsClassifier(n_neighbors = 5)
knn5.fit(X,y)
print(knn5.predict(X_new))


#------------------------------------------------------------------------------

# LOGISTIC REGRESSION

# Initiate the estimartor
logreg = LogisticRegression()

# Fit the model with data
logreg.fit(X,y)

# Predict the response for the new data
print(logreg.predict(X_new))


#------------------------------------------------------------------------------

# COMPARE MODELS: TRAINING ACCURACY

# predict your training data
y_pred = logreg.predict(X)
y_pred1 = knn1.predict(X)
y_pred5 = knn5.predict(X)

# cross check that nothing got lost
print(len(y_pred))

# proportion of accurate predictions
print("\nProportion of accurate predictions in the traing data:")
print("Logistic regression: "); print(metrics.accuracy_score(y, y_pred))
print("KNN with k = 5:"); print(metrics.accuracy_score(y, y_pred5))

print("\nPredicting the training data with k = 1 reproduces the data")
print("KNN with k = 1:"); print(metrics.accuracy_score(y, y_pred1)); print("")


#------------------------------------------------------------------------------

# COMPARE MODELS: TRAIN/ TEST SPLIT

# split the data in training and test set
# notice that the splitting method has been deprecated and will be removed 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4)
print(X_test.shape); print(y_test.shape)


logreg.fit(X_train,y_train)
knn1.fit(X_train,y_train)
knn5.fit(X_train,y_train)


# predict your training data
y_pred = logreg.predict(X_test)
y_pred1 = knn1.predict(X_test)
y_pred5 = knn5.predict(X_test)

# cross check that nothing got lost
print(len(y_pred))

# proportion of accurate predictions
print("\nProportion of accurate predictions in the traing data:")
print("Logistic regression: "); print(metrics.accuracy_score(y_test, y_pred))
print("KNN with k = 5:"); print(metrics.accuracy_score(y_test, y_pred5))

print("\nThis also works with k = 1")
print("KNN with k = 1:"); print(metrics.accuracy_score(y_test, y_pred1)); print("")


#------------------------------------------------------------------------------

# CHOOSING K FOR KNN

# Try K = 1 through K = 25 and record accuracy
k_range = range(1, 26); scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train); y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

# Draw a figure to illustrate the results
plt.plot(k_range, scores); plt.xlabel('Value of K for KNN'); plt.ylabel('Testing Accuracy')
plt.show()

#------------------------------------------------------------------------------

# MAKING PREDICTIONS ON OUT-OF-SAMPLE DATA

# Initiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors = 11)

# Train the model with X and y
knn.fit(X,y)

# Make prediction for an out-of-sample observation
print(knn.predict([[3, 5, 4, 2]]))


#------------------------------------------------------------------------------

# K-FOLD CROSS VALIDATION

# Simulate splitting a dataset of 25 observations into 5 folds
kf = KFold(25, n_folds = 5, shuffle = False)

# print the contents of each each training and test set
print('{:^6} {:^50} {:^15}'.format('Iteration', 'Training set observations', 'Testing set observations'))

for iteration, data in enumerate(kf, 1):
    print('{:^6} {} {:^15}'.format(iteration, str(data[0]), str(data[1])))

print("")

# 10-fold cross-validation with K = 5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors = 5)
scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')

# Use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())

# search for optimal value of K for KNN
k_range = range(1,31); k_scores = []; 

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    k_scores.append(scores.mean())

print(k_scores)

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores); plt.xlabel('Value of K for KNN'); plt.ylabel('Cross-Validated Accuracy')
plt.show()
print("")


#------------------------------------------------------------------------------

# MODEL SELECTION: COMPARE THE BEST KNN MODEL WITH LOGISTIC REGRESSION

# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors = 20)
print(cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy').mean())
print("")

# 10-fold cross-validation with logistic regression
print(cross_val_score(logreg, X, y, cv = 10, scoring = 'accuracy').mean())
print("")


#------------------------------------------------------------------------------

# MORE EFFICIENT PARAMETER TUNING USING GridSearchCV

param_grid = dict(n_neighbors = list(k_range))        # using here same k_range than earlier
print(param_grid)

# initiate the grid
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')

# fit the grid with data
grid.fit(X, y)

# view the complete results (list of named tuples)
grid.grid_scores_

# examine the first tuple
print("\nParameters:")
print(grid.grid_scores_[0].parameters)
print("\nCV validation scores:")
print(grid.grid_scores_[0].cv_validation_scores)
print("\nMean validation scores:")
print(grid.grid_scores_[0].mean_validation_score)
print("")

# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)

# plot the results
plt.plot(k_range, grid_mean_scores); 
plt.xlabel('Value of K for KNN'); plt.ylabel('Cross-Validated Accuracy'); plt.show()

# examine the best model
print(grid.best_score_); print(grid.best_params_); print(grid.best_estimator_)
print("")

#------------------------------------------------------------------------------

# SEARCHING MULTIPLE PARAMETERS SIMULTANEOUSLY

# define the parameter values to be searched
k_range = list(range(1,31)); weight_options = ['uniform', 'distance'] # distance: close neighbors weighted more heavily

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors = k_range, weights = weight_options)
print(param_grid)
print("")

# intantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
grid.fit(X,y)

# view the complete results
print(grid.grid_scores_)
print("")

# examine the best model
print(grid.best_score_); print(grid.best_params_)
print("")

# refit the best model using all of the data to make a prediction on out-of-sample data
print(grid.predict([[3, 5, 4, 2]]))


#------------------------------------------------------------------------------

# REDUCING COMPUTATIONAL EXPENSE USING RANDOMIZED SEARCH

# specify parameter distributions rather than parameter grid
param_dist = dict(n_neighbors = k_range, weights = weight_options)
        # Important: specify a continuous distribution (rather than a list of values) for any cont params

rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring = 'accuracy', n_iter = 10,
                          random_state = 5)        # n_iter controls number of searches
rand.fit(X, y); print(rand.grid_scores_)
print("")

# run randomized search 20 times and recod the best score
best_scores = []

for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring = 'accuracy', n_iter = 10)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
    
print(best_scores)








