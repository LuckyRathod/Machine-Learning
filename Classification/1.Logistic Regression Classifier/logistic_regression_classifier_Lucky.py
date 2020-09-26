# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:55:01 2018

@author: Lucky_Rathod
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
'''
Now here we will be using Age and Salary of Employee[Independent varaible]
to predict the probability that whether he/she will PURCHASE[Dependent variable] the product[Car]

'''
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
#we need to apply feature scaling because the value of salary and age is not in same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#Fitting Logistic regression to training set

'''
Logistic Regression

When a sigmoid function is used in Linear regression formula we get Logistic regression formula
Trend Line in Logistic regression Classifier is same as Trend line in Linear Regression 
It follows the formula to get the BEST FITTING line that can fit the dataset 
Generally it is used to predict the probability [i-e Probability of whether the user will buy product or not]

'''

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#Predicting the probability that whether the customer will buy the product or not
y_pred = classifier.predict(X_test)
#Compare y_pred and y_test


#Evaluating the performance by Confusion Matrix
'''
Confusion matrix contains correct and incorrect prediction

'''

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#1st param -> Truth value of data 
#2nd param -> Predicted value of data
#See in console -> array([[65,3],[8,24]])
#11 Incorrect 89->Correct 


#Graphical visualization
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'''
Still we can see that in above Diagram some green points are in Red region
and some Red points are in green region ->INCOREECT PREDICTIONS

Boundary line is the BEST fitting line that Logistic classifier founded

GOAL -> Classify the Right user into Right category

So that if new points arrives then we can directly predict
->If point lies in Red Region -> Wont buy the product
->If point lies in Green Region -> Will buy the product


'''



# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()












