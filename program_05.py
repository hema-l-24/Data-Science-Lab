#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


iris = load_iris()
X = iris.data
y = iris.target


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


kernels = ['rbf']
gammas = [0.5]
Cs = [0.01, 1, 10]
best_accuracy = 0
best_parameters = None
best_support_vectors = None


# In[5]:


for kernel in kernels:
    for gamma in gammas:
        for C in Cs:
            svm_clf = SVC(kernel=kernel, gamma=gamma, C=C, decision_function_shape='ovr')
            svm_clf.fit(X_train, y_train)
            y_pred = svm_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            total_support_vectors = np.sum(svm_clf.n_support_)
            print(f"Kernel: {kernel}, Gamma: {gamma}, C: {C}, Accuracy: {accuracy}, Total Support Vectors: {total_support_vectors}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_parameters = (kernel, gamma, C)
                best_support_vectors = total_support_vectors


# In[6]:


print("\nBest Classification Accuracy:", best_accuracy)


# In[7]:


print("Best Hyperparameters:", best_parameters)


# In[8]:


print("Total Support Vectors for Best Model:", best_support_vectors)

