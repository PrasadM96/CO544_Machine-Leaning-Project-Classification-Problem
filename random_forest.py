#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#read the train dataset
dataset = pd.read_csv('trainData.csv')
dataset.head()

#read the test dataset
testDataset=pd.read_csv('testdata.csv')
testDataset.head()


#handling missing values in train data
dataset.replace({"?":np.nan},inplace=True) 
dataset = dataset.apply(lambda x: x.fillna(x.value_counts().index[0]))

#handling missing values in test data
testDataset.replace({"?":np.nan},inplace=True)
testDataset=testDataset.apply(lambda x: x.fillna(x.value_counts().index[0]))


# In[17]:


#convert nominal attributes to numeric attributes
lb_make = LabelEncoder()
l=['A1','A3','A4','A6','A8','A9','A11','A13','A15']

for i in l:
     temp = lb_make.fit_transform(dataset[i])
     dataset[i] = temp

     temp1 = lb_make.fit_transform(testDataset[i])
     testDataset[i] = temp1


X = dataset.drop('A16' , axis='columns')
y = dataset['A16']

#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state = 101)


# In[20]:


from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=3, min_samples_split=2, 
                                            min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, 
                                            oob_score=False, n_jobs=1, random_state=None, verbose=0)
# model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
# accuracy_score(y_test,y_pred)

model.fit(X,y)
y_pred=model.predict(testDataset)

print('\n'.join(y_pred))


# In[ ]:





# In[ ]:




