# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 20:30:08 2018

@author: Lucky_Rathod
"""
# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
# Create your regressor here
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

# Predicting a new result

#Average value of 6.5 which comes in the interval of Decision tree is 150k
y_pred = regressor.predict(6.5) #Ans 150k

# Visualising the Regression results
'''
In these , we are plotting independent variable at X dependent varaible at y
Decision tree splits and in each Leaf it gives average value of all points to new variable that arrives

Here we only have 10salary and 10 levels
It joins its prediction by straight line because it had no predictions to plot


Decision tree is a Non Continous(Both Horizontal and Vertical Lines) Non Linear Regression Model



'''
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


'''
To make you understand better plot the graph with higher resolution 
Because there will be 90 values
It will have both horizontal and vertical line 
So any value that comes in That interval will have average varaible

Each point will be in between each interval see the diagram 

'''

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()