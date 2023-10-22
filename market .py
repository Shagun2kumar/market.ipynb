#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[2]:


df=pd.read_csv("Mall_Customers.csv")
df.head(10)


# In[3]:


df.shape


# In[4]:


#to check null value
df.info()


# In[5]:


df.describe()


# In[6]:


x=df.iloc[:,[3,4]].values
x


# In[7]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.boxplot(y=df["Spending Score (1-100)"], color="yellow")
plt.subplot(1,2,2)
sns.boxplot(y=df["Annual Income (k$)"])
plt.show()


# In[8]:


genders = df.Gender.value_counts()
sns.set_style("whitegrid")
plt.figure(figsize=(14,6))
sns.barplot(x=genders.index, y=genders.values)
plt.show()


# In[9]:


#wcss stands for within cluster sum of square
#kmeans.inertia_ is a formula  for cluster
wcss=[]
for  i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('no. of cluster')
plt.ylabel('wcss values')
plt.show()


# In[10]:


#building model
kmeansmodel=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeansmodel.fit_predict(x)


# In[11]:


kmeansmodel=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeansmodel.fit_predict(x)
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=80,c="blue",label='customer1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=80,c="red",label='customer2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=80,c="green",label='customer3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=80,c="yellow",label='customer4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=80,c="orange",label='customer5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="magenta",label='centroids')

plt.title("clusters of customers")
plt.xlabel("annual income")
plt.ylabel("spending income")
plt.legend()
plt.show()

