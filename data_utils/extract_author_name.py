#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd

d1 = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\Data\final_data.csv")
d2 = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\Data\data_23.csv")

# In[18]:


d1

# In[19]:



print(d1.columns)
print(d2.columns)

# In[21]:


d2

# In[14]:


# Ensure keys are the same type
d1['ha_id'] = d1['ha_id'].astype(str)
d2['author_id'] = d2['author_id'].astype(str)

# Create lookup dictionaries from d2 for all columns we want to add
author_map = dict(zip(d2['author_id'], d2['author_name']))
time_map = dict(zip(d2['author_id'], d2['time']))
weekday_map = dict(zip(d2['author_id'], d2['weekday']))
month_map = dict(zip(d2['author_id'], d2['month']))
year_map = dict(zip(d2['author_id'], d2['year']))

# Map all columns to d1
d1['author_name'] = d1['ha_id'].map(author_map)
d1['time'] = d1['ha_id'].map(time_map)
d1['weekday'] = d1['ha_id'].map(weekday_map)
d1['month'] = d1['ha_id'].map(month_map)
d1['year'] = d1['ha_id'].map(year_map)

# In[15]:


d1

# In[16]:


d1.to_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\final_data_with_author_names.csv", index=False)

# In[ ]:



