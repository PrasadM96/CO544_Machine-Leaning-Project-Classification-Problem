#import libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

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

#convert nominal attributes to numeric attributes
lb_make = LabelEncoder()
l=['A1','A3','A4','A6','A8','A9','A11','A13','A15']
for i in l:
     temp = lb_make.fit_transform(dataset[i])
     dataset[i] = temp

     temp1 = lb_make.fit_transform(testDataset[i])
     testDataset[i] = temp1
    
#x and y
X = dataset.drop('A16' , axis='columns')
y = dataset['A16']

#base classifiers
estimators = [
    ('rf', RandomForestClassifier(n_estimators=2000, random_state=42)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42,max_iter=2000))) ]
#stacking
clf = StackingClassifier(
     estimators=estimators,final_estimator=LogisticRegression(),  cv=3, stack_method='auto', n_jobs=None, 
    passthrough=False, verbose=0
 )

#train model
#clf.fit(X_train,y_train)
#predict for test data
#y_pred=clf.predict(X_test)
#calculate accuracy
#accuracy_score(y_test,y_pred)


#train model
clf.fit(X,y)
#predict for test data
y_pred=clf.predict(testDataset)
print('\n'.join(y_pred))


