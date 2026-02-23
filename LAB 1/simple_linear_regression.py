#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


wine_data = pd.read_csv("/Users/rohni./Documents/ML/LAB 1/WineQT.csv")


# In[ ]:


print(wine_data.head())
print(wine_data.shape)
wine_data.info()


# In[ ]:


wine_data.isnull().sum()


# In[ ]:


X_alcohol = wine_data[['alcohol']]
y = wine_data['quality']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X_alcohol, y, test_size=0.2, random_state=42
)


# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)
y_pred


# In[ ]:


print(model.score(X_test, y_test))


# In[ ]:


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)


# 

# In[ ]:


plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Alcohol")
plt.ylabel("Wine Quality")
plt.title("Best Fit Line: Alcohol vs Wine Quality")
plt.show()

