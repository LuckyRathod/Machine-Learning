# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:43:28 2018

@author: Lucky_Rathod
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Importing data set
dataset = pd.read_csv('50_Startups.csv')

#Splitting data set into independent and dependent matrix
X = dataset.iloc[:,:-1].values
#dependent matrix
y= dataset.iloc[:,4].values

#Categorical variable
#Now ml models have equations which only have values and not text so we need to convert or encode to values of text

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder();
X[:,3] = labelencoder_X.fit_transform(X[:,3])

#now text is converted to values 0,2,1
#But ml model will think 2>1>0
#but we have category relationship between these variables

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()


#Avoid Dummy variable trap
#Here we are removing first column 
#Because we know we have to remove one dummy variable
X = X[:,1:]


#Train test split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)



"""#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""


#Now we will build  our Multiple linear regression model to our training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting test set results
y_pred  = regressor.predict(X_test)



#Building an optimal team of independent variables
import statsmodels.formula.api as sm
X= np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)


#Backward Elimination starts from here
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

"""
    See the X variable to choose correct index to be removed
    

"""

#Remove index variable 2 from X because it  has p value > SL
#See the X variable in variable explores to remove correct index no

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


#Train test split
from sklearn.cross_validation import train_test_split
X_trainOptimal,X_testOptimal,y_trainOptimal,y_testOptimal = train_test_split(X_opt,y,test_size=0.2,random_state=0)



"""#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""


#Now we will build  our Multiple linear regression model to our training set
from sklearn.linear_model import LinearRegression
regressorOptimal = LinearRegression()
regressorOptimal.fit(X_trainOptimal,y_trainOptimal)

#Predicting test set results
y_predOptimal  = regressorOptimal.predict(X_testOptimal)





