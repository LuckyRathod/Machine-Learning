# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:56:57 2018

@author: Lucky_Rathod
"""

#Multi Armed Bandit Problem

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
'''
Car company prepared an 10 Ads that they would put on Social Network
They are not sure about which ads to be put to gain maximum clicks
Create a best strategy to find out which ads to be displayed    

Each time user login we will one of 10 ads and if user clicks we will get reward as 1 or else 0
So we have a dataset of 10,000 users

Which version of ad will be displayed will depend on results of previous rounds 
    -> These is ReInforcement or Online or Interactive Learning
    -> Because strategy is Dynamic 
    
If we randomly select then total_rewards = 1200 (Random Selection Algorithm)
'''

#Implementing UCB
import math
d=10
number_of_selection = [0] * d #Create a vector that will contain no of selection of each ad i
sum_of_rewards = [0] * d
N=10000

ads_selected=[] #Ads version that were selected in each round
total_reward=0
#Now we going through all rounds from 0 to 10000
for n in range(0,N):
    ad=0
    max_upper_bound = 0 
    #Compute  Average reward of ad-i upto round n and Confidence interval
    #For each version of ad-i
    for i in range(0,d):
        if(number_of_selection[i]>0): 
            #Now here we are at specific round Dealing with specific version of ad 
            average_reward = sum_of_rewards[i]/number_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/number_of_selection[i])
            upper_bound =average_reward + delta_i
        else:
             #At round=0 we will enter in these part
             '''
             For first 10rounds we make random selecion and then we use above if strategy for maximum return
             At round 0 = ad 0 will be selected
             At round 1 = ad 1 will be selected
             At round 2 = ad 2 will be selected and so on till 10
             
             
             '''
             upper_bound=1e400
        #Select ad-i that has maximum ucb
        #if upper_bound is high than max_upper_bound the max_upper_bound = upper_bound
        if upper_bound>max_upper_bound:
            max_upper_bound = upper_bound
            ad=i
    ads_selected.append(ad)
    number_of_selection[ad]=number_of_selection[ad]+1 #No of times each ad was selected uptill n round
    reward = dataset.values[n,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward   
    
#At last rounds you will see that there is Ad 4 which is selected most of the time
#Because ucb strategy found that AD4 is best.See ads selected variable and see last 10 rounds
#total reward by ucb ->2178
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
    
    
    
    
    
