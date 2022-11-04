#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics

# In[18]:


df = pd.read_csv("USA_Housing.csv")
#df


# In[19]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[24]:


from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 

lm.fit(X_train,y_train)


# In[25]:


predictions = lm.predict(X_test) 


# In[28]:


#plt.scatter(y_test,predictions)


# In[30]:




print('MAE:', metrics.mean_absolute_error(y_test, predictions)) 
print('MSE:', metrics.mean_squared_error(y_test, predictions)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[31]:



#pickle.dump(lm,open("model.pkl","wb"))


# In[32]:


model = pickle.load(open("model.pkl","rb"))
#model


# In[33]:


print(model.predict([[79545.458574,5.682861,7.009188,4.09,23086.800503]]))


# In[37]:


#X


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




