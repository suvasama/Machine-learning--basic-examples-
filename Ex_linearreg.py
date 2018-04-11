#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:08:19 2018

@author: suvasama
"""

#------------------------------------------------------------------------------

# IMPORT LIBRARIES

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression

# k-fold cross validation
from sklearn.cross_validation import cross_val_score

import numpy as np


#------------------------------------------------------------------------------

# IMPORT DATA 

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col = 0)

# display first and last 5 rows, shape of data frame
print(data.head()); print(data.tail()); print(data.shape)

# Draw scatter plots of the data with a regression line 
sns.pairplot(data,x_vars = ['TV', 'radio', 'newspaper'], y_vars = 'sales', size = 7, aspect = 0.7, kind = 'reg')
plt.show()      # prints the figure here

#------------------------------------------------------------------------------

# PREPARE THE DATA

X = data[['TV', 'radio', 'newspaper']]; y = data.sales
print(X.head()); print(y.head()); print(type(X)); print(type(y)); print(X.shape); print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)


#------------------------------------------------------------------------------

# RUN A LINEAR REGRESSION ON DATA

# initiate 
linreg = LinearRegression()

# fit into the training data
linreg.fit(X_train, y_train)

# display the results
print(linreg.intercept_)
coef_names = ['TV', 'radio', 'newspaper']; coefs = zip(coef_names, linreg.coef_); coefs = list(coefs)
print(coefs)
print("")


#------------------------------------------------------------------------------

# FEATURE SELECTION USING CROSS-VALIDATION

scores = cross_val_score(linreg, X, y, cv = 10, scoring = 'neg_mean_squared_error')
print("scores:")
print(scores)
print("")

# fix the sign of MSE scores
print("MSE scores:")
mse_scores = - scores; print(mse_scores)
print("")

# Convert MSE to RMSE
print("RMSE scores:")
rmse_scores = np.sqrt(mse_scores); print(rmse_scores)
print("")

# calculate the average RMSE
print("Mean of RMSE scores:")
print(rmse_scores.mean())
print("")

# 10-fold cross-validation with two features (excluding newspaper)
print("Mean of RMSE scores in the model without newspaper:")
feature_cols = ['TV', 'radio']; X = data[feature_cols]
print(np.sqrt(-cross_val_score(linreg, X, y, cv = 10, scoring = 'neg_mean_squared_error')).mean())
print("")








