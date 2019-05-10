# Importing Libraries
import numpy as np # for scientific computation
import matplotlib.pyplot as plt # for visualization
import pandas as pd # for data manuplation

# create dataset or points
from sklearn.datasets import make_blobs
x,y = make_blobs(n_samples=300, centers= 5, cluster_std=0.6)

# plotting the graph to get the insight
plt.scatter(x[:,0], x[:,1])
plt.show()

# Importing unsuperised K means clustering algorithm
from sklearn.cluster import KMeans

# creating list of within cluster variable
wcv = []

# Using loop 
for i in range(1,11):
    km = KMeans(n_clusters = i)
    km.fit(x)
    wcv.append(km.inertia_)
    
plt.plot(range(1,11), wcv)
plt.show()

# creating no of cluster and y prediction
km = KMeans(n_clusters = 5)
y_predict = km.fit_predict(x)

# Plotting the graph b/w cluster and points
plt.scatter(x[y_predict == 0, 0], x[y_predict == 0,1], c = "r", label = "Smart Customers")    
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1,1], c = "b", label = "Target Customers")    
plt.scatter(x[y_predict == 2, 0], x[y_predict == 2,1], c = "g", label = "Smart Customers")    
plt.scatter(x[y_predict == 3, 0], x[y_predict == 3,1], c = "y", label = "Target Customers")    
plt.scatter(x[y_predict == 4, 0], x[y_predict == 4,1], c = "magenta", label = "Smart Customers")

plt.legend()
plt.show()

    