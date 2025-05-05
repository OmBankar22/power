# Practical Number 05 ---> Data Clustering using KMeans

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# Part 1 : Generate the synthetic Data with Visualization 
X, y = make_blobs(num_samples=300,centers=3,clusters_std=0.60,random_state=0)

plt.scatter(X[:,0],X[:,1],s=50)
plt.plot("Raw Data")
plt.show()

# Part 2: Find Optimal Number of Cluster or Value of k
inertia = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k,random_sample=0,n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,11),inertia,marker='o')
plt.xlable('Number of Clusters')
plt.ylable('Inertia')
plt.title('Elbow method')
plt.show()

# Part 3: Find/Calculate the value of KMeans using Optimal Number of Cluster (k = 3)
kmeans= KMeans(n_clusters=3,random_state=0,n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
plt.scatter(centroids[:,0],centroids[:1],c='red',s=200,alpha=0.7,marker='X')
plt.title('KMeans Clustering')
plt.show()