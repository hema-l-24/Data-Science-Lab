#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


# In[2]:


data = np.loadtxt('spiral.txt',skiprows=1)


# In[3]:


X = data[:, :2]
true_labels = data[:, 2]


# In[4]:


# Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', s=50, alpha=0.8)
plt.title("True Clusters")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# In[5]:


# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_pred = kmeans.fit_predict(X)
kmeans_rand_index = adjusted_rand_score(true_labels, kmeans_pred)
print("Rand Index for K-means:", kmeans_rand_index)


# In[6]:


# Single-link Hierarchical Clustering
single_link = AgglomerativeClustering(n_clusters=3, linkage='single')
single_link_pred = single_link.fit_predict(X)
single_link_rand_index = adjusted_rand_score(true_labels, single_link_pred)
print("Rand Index for Single-link Hierarchical Clustering:", single_link_rand_index)


# In[7]:


# Complete-link Hierarchical Clustering
complete_link = AgglomerativeClustering(n_clusters=3, linkage='complete')
complete_link_pred = complete_link.fit_predict(X)
complete_link_rand_index = adjusted_rand_score(true_labels, complete_link_pred)
print("Rand Index for Complete-link Hierarchical Clustering:", complete_link_rand_index)


# In[8]:


# Visualize the clustering results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_pred, cmap='viridis', s=50, alpha=0.8)
plt.title("K-means Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=single_link_pred, cmap='viridis', s=50, alpha=0.8)
plt.title("Single-link Hierarchical Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=complete_link_pred, cmap='viridis', s=50, alpha=0.8)
plt.title("Complete-link Hierarchical Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

