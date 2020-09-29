# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:43:56 2018

@author: Lucky_Rathod
"""
#k-means Clustering

'''
It allows you to cluster your data

How it works ?

1. Choose the number k of clusters Assume k=2
2. Select random k(2) points which will be your centroids of cluster.Random points should not neccassarily from dataset
3. Assign each data point to closest centroid through Euclidean Distance
4. Compute and place the new centroid of each cluster [Find the center of cluster which will be your new centroids]
5. ReAssign each data point to new closest centroid 
   If any Re-Assignment took place Goto step 4 otherwise FINISH

'''

#Importing the libraries
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Data set
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

'''
Within Cluster Sum of Square ->Its a list which will contain wcss graph values
Optimal no of cluster is calculated when the line in the graph becomes constant

'''
wcss = [] #wccss is also called as INERTIA

'''
To find optimal no  of cluster we will be using Elbow method

'''
for i in range(1, 11):
    
    '''
    In these loop we will do two things
    #1. Fit k-means to X
    #2. Append wcss value to wcss list
    
    '''
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter=300,n_init=10,random_state = 0)
    kmeans.fit(X) #1 
    wcss.append(kmeans.inertia_) #2   
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset

    '''
    To prevent Random Initialization trap we use k-means++
    If we select random centroids at intial step its hard to find the perfect cluster
    
    '''
kmeans = KMeans(n_clusters = 5, init = 'k-means++',max_iter=300,n_init=10,random_state = 0)

'''
init     -> Selection Intialization points for cluster -> Through k-means++ ,random can also be used but Random Initialization trap 
max_iter -> Maximum no of iteration to find Final Clusters. Default -> 300
n_init   -> No of times kmeans algorithm will run with different intial centroids

'''
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
'''
X[y_kmeans==0,0] -> we take X-co-ordinate of all observation points that belong to cluster 1 
X[y_kmeans==0,1] -> we take Y-co-ordinate of all observation points that belong to cluster 1 
s -> Size
'''
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()