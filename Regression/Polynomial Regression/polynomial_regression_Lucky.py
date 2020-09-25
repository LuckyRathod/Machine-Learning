# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:36:18 2018

@author: Lucky_Rathod
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Importing data set
dataset = pd.read_csv('Position_Salaries.csv')

#Splitting data set into independent and dependent matrix
X = dataset.iloc[:,1:2].values  #2 is excluded
#dependent matrix
y= dataset.iloc[:,2].values

#Train test split
#No need to split because dataset is small

"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)"""

"""Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""


#Linear Regression model
from sklearn.linear_model import LinearRegression
regressorL = LinearRegression()
regressorL.fit(X,y)

#Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures
#Higher the degree better the prediction
regressorP = PolynomialFeatures(degree=4) #default 2
""" X will be transformed to X_poly which not only contains independent variable 
    But also Polynomial value X^2 X^3 .......
    Degree=2 means indpendent variable and X^2
    Matrix also contains b0 constants in first column [All rows have value 1]"""
X_poly = regressorP.fit_transform(X)

"""
    X_poly will be fitted to object of Linear Regresson model

"""
regressorL2 = LinearRegression()
regressorL2.fit(X_poly,y)


# Visualization Linear Regression
#For data points
plt.scatter(X,y,color='red')
#For regression line
plt.plot(X,regressorL.predict(X),color='blue')
plt.title('Linear Regression Model Results - Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()
#Salary at 6.5 level ->330k



#Visualization Polynomial Regression - Degree 2
#For data points
plt.scatter(X,y,color='red')
#For regression line
'''
Now here you will replace regressorL with regressorL2 and X with X_poly
But X_poly is already defined 
Better Genalization should be maintained for future models
Replace X with regressorP.fit_tranform(X) 

'''
plt.plot(X,regressorL2.predict(regressorP.fit_transform(X)),color='blue')
plt.title('Polynomial Regression (Degree 2) Model Results - Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()


#Visualization Polynomial Regression - Degree 4   higher Accuracy 

'''
To get a continous curve and to avoid straight lines
Increase the dataset X values with 0.1 
An then give that X values as an input to graph

'''

X_continous_curve = np.arange(min(X),max(X),0.1)
#To make X_continous a matrix 
X_continous_curve = X_continous_curve.reshape((len(X_continous_curve),1))
#For data points
plt.scatter(X,y,color='red')
#For regression line
'''
Now here you will replace regressorL with regressorL2 and X with X_poly
But X_poly is already defined 
Better Genalization should be maintained for future models
Replace X with regressorP.fit_tranform(X) 

'''
plt.plot(X_continous_curve,regressorL2.predict(regressorP.fit_transform(X_continous_curve)),color='blue')
plt.title('Polynomial Regression (Degree 4) Model Results - Truth or Bluff')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()


#Prediction of Truth or bluff at level 6.5 --> Ans said by the new employee was 160k

#Linear model prediction
regressorL.predict(6.5) #Ans 330k

#Polynoial regressor prediction at degree 4 and continous curve
regressorL2.predict(regressorP.fit_transform(6.5)) #Ans 158.886 k  -> TRUTH




