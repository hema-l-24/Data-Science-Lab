#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
hours_studied=[10,8,5,3,6,2]
exam_scores=[95,86,82,98,94,92]
plt.plot(hours_studied,exam_scores,marker='*',color='red',linestyle='-')
plt.xlabel('Hours Studied')
plt.ylabel('Score in final exam')
plt.title('Effect of Hours Studied on Exam Score')
plt.show()


# In[ ]:




