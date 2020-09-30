# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:05:20 2018

@author: Lucky_Rathod
"""

# Hierarchical Clustering
'''
1. Make each data point a single point cluster 
2. Take the two closest datapoints and make them one cluster ->N-1 Clusters
3. Take the two closest cluster and make them one cluster    ->N-2 Clusters
4. Repeat step 3 untill there is only one cluster 

Hierarchical Clustering maintains a memory of how we went through above process
And that memory is stored in Dendrogram

'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

#Using Dendrogram to find optimal no of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward')) #ward -> Minimize the variance(dis-similarity)in each cluster 
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

'''
From Dendrogram select the Longest Vertical line that does not cross any Horizontal line
Once the Longest vertical line is selected, Apply the DIS-SIMILARITY THRESOLD LINE which cross vertical line selected
After applying the thresold line count the no of vertical lines which are just above Thresold line
These no of vertical lines will be your total no of cluster 

'''

#Fitting Hierarchical clustering to X
from sklearn.cluster import AgglomerativeClustering #Agglomerative Clustering -> Bottom up Approach
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')#affinity -> distance to do linkage 
y_hc= hc.fit_predict(X)


# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()