# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:25:05 2018

@author: Lucky_Rathod
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Importing data set
dataset = pd.read_csv('Salary_Data.csv')

#Splitting data set into independent and dependent matrix
X = dataset.iloc[:,:-1].values
#dependent matrix
y= dataset.iloc[:,1].values

#Train test split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

"""Feature Scaling
No need for Feature Scaling in Simple linear regression model

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
"""

#Fitting simple linear regresson on Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Our model has learn the correlation
#now we will predict the X_test set values
#which will then be compared by y_test

y_pred = regressor.predict(X_test)


#Visualization of the prediction values


###########    Training Set    #############
#Plotting the real values [Salary and experiance] of employees
plt.scatter(X_train,y_train,color='red')
#Plotting Regression line i-e Predicting values [Salary corresponding to its experiance]
plt.plot(X_train,regressor.predict(X_train),color='blue')

plt.title('Salary vs Experiance [Training Set]')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()



##########    Testing Set    #############
#Plotting the real values [Salary and experiance] of employees
plt.scatter(X_test,y_test,color='red')
#Plotting Regression line i-e Predicting values [Salary corresponding to its experiance]

#when we trained the model we obtained unique model equation which is same and generates same 
#regression line everytime
#so if we dont change the below line in Test set its fine , even if we change its fine
plt.plot(X_train,regressor.predict(X_train),color='blue')

plt.title('Salary vs Experiance [Testing Set]')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()




