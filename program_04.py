#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


iris = load_iris()
X = iris.data
y = iris.target


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


log_reg = LogisticRegression(C=1e4, penalty='l2', solver='lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)


# In[5]:


y_pred = log_reg.predict(X_test)


# In[6]:


accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)

