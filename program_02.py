#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


get_ipython().system('dir mtc*')


# In[13]:


get_ipython().system('cd')


# In[2]:


df=pd.read_csv("mtcars.csv")


# In[3]:


df.shape


# In[27]:


df.head()


# In[6]:


df.info()


# In[5]:


df.describe()


# In[7]:


df.columns


# In[21]:


df.dtypes


# In[22]:


df.mpg.describe


# In[26]:


plt.hist(df['mpg'],bins=10,color='blue',edgecolor='black')
plt.xlabel('Miles per gallon(mpg)')
plt.ylabel('Frequency')
plt.title('Histogram of Miles per gallon(mpg)')
plt.show()

