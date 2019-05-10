# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# creating dataset
from sklearn.datasets import make_blobs
x,y = make_blobs(n_samples=300, centers= 5, cluster_std=0.6)

#plotting to get insight
plt.scatter(x[:,0], x[:,1])
plt.show()

# creating dendogram
import scipy.cluster.hierarchy as sch
sch.dendrogram(sch.linkage(x))

# Importing hierarchical cluster algorithm
from sklearn.cluster import AgglomerativeClustering
hca = AgglomerativeClustering(n_clusters= 5)
y_predict = hca.fit_predict(x)

# plotting to get insight
plt.scatter(x[y_predict == 0, 0], x[y_predict == 0,1], c = "r", label = "Smart Customers")    
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1,1], c = "b", label = "Target Customers")    
plt.scatter(x[y_predict == 2, 0], x[y_predict == 2,1], c = "g", label = "Smart Customers")    
plt.scatter(x[y_predict == 3, 0], x[y_predict == 3,1], c = "y", label = "Target Customers")    
plt.scatter(x[y_predict == 4, 0], x[y_predict == 4,1], c = "magenta", label = "Smart Customers")

plt.legend()
plt.show()

    