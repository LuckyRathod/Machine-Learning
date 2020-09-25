# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:29:35 2018

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


'''
Random forest is a non linear regression model 
It is team of decision trees
Each one making predictions of your dependent varaibles
and its output is average of all these predictions

'''

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

# Predicting a new result
'''
As no of trees increases predictions gets accurate 

As before we want our prediction close to 160k

10 trees -> 167k
100 trees -> 158300
300 trees -> 160300

'''
y_pred = regressor.predict(6.5) 



# Visualising the Regression results (for higher resolution and smoother curve)

'''
It doesnt means that we will get more stairs as we increase our no of trees
Random forest will choose and place the stairs  accurately 
And you will get better precitions

'''


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()