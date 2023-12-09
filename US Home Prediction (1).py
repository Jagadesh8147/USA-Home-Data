#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


data=pd.read_csv('/Users/lucky/Downloads/USA_Housing.csv')


# In[4]:


data


# In[5]:


data.head(100)


# In[6]:


data.tail(100)


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.isna().sum()


# In[10]:


data.shape


# In[11]:


data1=data.drop('Address',axis=1)


# In[12]:


data1


# In[14]:


data2=pd.get_dummies(data1,dtype=int)


# In[15]:


data2


# In[16]:


y=data['Price']


# In[17]:


y


# In[18]:


x=data2.drop('Price',axis=1)


# In[19]:


x


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=40)


# In[22]:


x_train.shape


# In[23]:


y_train.shape


# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


lreg=LinearRegression()


# In[26]:


lreg.fit(x_train,y_train)


# In[27]:


ypred=lreg.predict(x_test)


# In[28]:


ypred


# In[29]:


from sklearn.metrics import r2_score


# In[31]:


r2_score(ypred,y_test)


# In[32]:


from sklearn.metrics import mean_squared_error


# In[33]:


mean_squared_error(ypred,y_test)


# In[34]:


import seaborn as sns


# In[38]:


import matplotlib.pyplot as py


# In[ ]:




