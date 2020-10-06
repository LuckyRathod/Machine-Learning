# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:41:14 2018

@author: Lucky_Rathod
"""
#Associate rule Learning  -> Example : Movie Recommended System 
#Apriori 
'''
People who will buy this will also buy this

1.Support(I)         -> Transactions containing I/Total no of transaction
2.Confidence(I1->I2) -> Transactions containing I1 & I2 / Transaction containing I1 
                        It says that if user buys I1 then confidence value is the probability
                        that it will also buy I2
3.Lift(I1->I2)       -> Confidence(I1->I2)/Support(I2) 
                        Lift is improvement in our Prediction
                        If we target the users who bough I1 will have greater impact (17.5%)then
                        If we target only the users who bought I2(10%)
                        So Lift is 17.5/10 ->1.75%
                        
Alogrithm
1. Set minimum support and confidence
2. Take all subsets in transactions having higher support than minimum support
3. Take all rules of these subsets having higher confidence then minimum confidence
4. Sort the rules by decreasing lift (Dont need in python)

'''

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
#If we dont write header=None then 1st row of dagtaset will be the column title of dataset 

'''
Input of Apriori algorithm is List of List means List containing Different Transaction
And each transaction elements will be in List

transaction is a list and transaction.append([]) represents list within list

'''

transaction=[]
for i in range(0,7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# Training Apriori on the dataset
'''
Now we will import apyori File which contains classes and libraries that are used to Build rules
apriori is function which have Following arguments
min_support    -> Example Items that are bought 3 or Four times a day ->3*7=21 time a week ->21/7500 -> 0.003
min_condidence -> If 0.8 means All rules has to be correct in 80% of time -> No rule will be the output -> 0.2
min_lift       -> Minimum Prediction improvement should be Higher than 3
'''
from apyori import apriori
rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

#In spyder you cant see the variables properly 
#Execute these code to know Rule , suppoet ,confidence and lift

for item in results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")