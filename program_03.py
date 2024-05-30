#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


books_df = pd.read_csv("BL-Flickr-Images-Book.csv")


# In[3]:


books_df.head()


# In[4]:


print("Original DataFrame:")
print(books_df.head())


# In[5]:


irrelevant_columns = ['Edition Statement', 'Corporate Author', 'Corporate Contributors', 'Former owner','Engraver', 'Contributors', 'Issuance type', 'Shelfmarks']
books_df.drop(columns=irrelevant_columns, inplace=True)


# In[6]:


books_df.set_index('Identifier', inplace=True)


# In[7]:


books_df['Date of Publication'] = books_df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)


# In[8]:


books_df['Date of Publication'] = pd.to_numeric(books_df['Date of Publication'], errors='coerce')


# In[9]:


print("\nCleaned DataFrame:")
print(books_df.head())


# In[ ]:




