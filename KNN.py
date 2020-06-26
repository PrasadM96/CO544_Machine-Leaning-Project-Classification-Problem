#!/usr/bin/env python
# coding: utf-8

# In[13]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('trainData.csv')
dataset.head()

testDataset=pd.read_csv('testdata.csv')
testDataset=pd.DataFrame(testDataset)
testDataset.head()

dataset.replace({"?":np.nan},inplace=True)
dataset = dataset.apply(lambda x: x.fillna(x.value_counts().index[0]))


testDataset.replace({"?":np.nan},inplace=True)
testDataset.fillna(method='ffill',axis=0,inplace=True)
#testDataset=testDataset.apply(lambda x: x.fillna(x.value_counts().index[0]))


#convert nominal to numeric
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
l=['A1','A3','A4','A6','A8','A9','A11','A13','A15']

for i in l:
     temp = lb_make.fit_transform(dataset[i])
     dataset[i] = temp

     temp1 = lb_make.fit_transform(testDataset[i])
     testDataset[i] = temp1

feature_names = ['A1','A2','A3','A4', 'A5','A6', 'A7','A8', 'A9','A10','A11',"A12","A13",'A14','A15']
X = dataset[feature_names]
y = dataset['A16']


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state = 101)


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X,y)

y_pred = model.predict(testDataset)

print('\n'.join(y_pred))


# In[ ]:




