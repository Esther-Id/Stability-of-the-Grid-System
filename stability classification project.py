#!/usr/bin/env python
# coding: utf-8

# # Stability of the Grid System

# In[1]:


#The values of the tp,tn,fp,fn

tp = 255
fp = 1380
fn = 45
tn = 20

precision = tp/(tp +fp)
recall = tp /(tp +fn)
f1_score = 2*(precision*recall)/ (precision + recall)

print("F1 Score is",round (f1_score,4))


# ### DATASET DESCRIPTIONS
# It has 12 primary predictive features and two dependent variables.
# 
# __Predictive features:__
# 
# - 'tau1' to 'tau4': the reaction time of each network participant, a real value within the range 0.5 to 10 ('tau1' corresponds to the supplier node, 'tau2' to 'tau4' to the consumer nodes);
# - 'p1' to 'p4': nominal power produced (positive) or consumed (negative) by each network participant, a real value within the range -2.0 to -0.5 for consumers ('p2' to 'p4'). As the total power consumed equals the total power generated, p1 (supplier node) = - (p2 + p3 + p4);
# - 'g1' to 'g4': price elasticity coefficient for each network participant, a real value within the range 0.05 to 1.00 ('g1' corresponds to the supplier node, 'g2' to 'g4' to the consumer nodes; 'g' stands for 'gamma');<br>
# __Dependent variables:__
# 
# - 'stab': the maximum real part of the characteristic differential equation root (if positive, the system is linearly unstable; if negative, linearly stable);
# - 'stabf': a categorical (binary) label ('stable' or 'unstable').
# 
# __Objective__
# - In this work, we’ll build a binary classification model to predict if a grid is stable or unstable using the UCI Electrical Grid Stability Simulated dataset.

# In[2]:


# Importing libraries
import pandas as pd
import numpy as np

import matplotlib as plt
import seaborn as sns


# In[3]:


#loading the dataset
df = pd.read_csv('Data_for_UCI_named.csv')
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


# Checking for missing values
df.isnull().sum()


# In[7]:


# Having a look at the target column

df['stabf'].value_counts()

sns.countplot(x = "stabf", data = df)


# In[8]:


df.columns


# In[9]:


# Dropping the stab column as direccted 
df.drop('stab', axis = 1, inplace = True)
df


# In[10]:


#Seperating the X and Y features
X = df.drop(columns='stabf')
y = df['stabf']
y.head()


# In[11]:


#split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_train.value_counts()


# In[12]:


# Replacing string values to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_test


# In[13]:


# Scaling the dataset using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, columns  = x_train.columns)



x_test = x_test.reset_index(drop = True)
x_test_scaled = scaler.fit_transform(x_test)

x_test_scaled = pd.DataFrame(x_test_scaled,columns =  x_test.columns)
x_train_scaled 


# In[14]:


x_train_scaled.shape,x_test_scaled.shape 


# ### Model building

# In[15]:


# Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(random_state=1)
rnd_clf.fit(x_train_scaled, y_train)
y_pred_rf = rnd_clf.predict(x_test_scaled)


# In[16]:


from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score
#classification report
print(classification_report(y_test, y_pred_rf, digits=4))


# In[17]:


print("Accuracy Score is",round (accuracy_score(y_test, y_pred_rf),4))


# In[18]:


# Xgboost classifier
from xgboost import XGBRFClassifier
extreme = XGBRFClassifier(random_state=1)
extreme.fit(x_train_scaled, y_train)
extreme_pred = extreme.predict(x_test_scaled)


# In[19]:


#classification report
print(classification_report(y_test, extreme_pred, digits=4))


# In[20]:


#The accuracy on the test
print( accuracy_score(y_test, extreme_pred))


# In[21]:


#Lightgbm
from lightgbm import LGBMClassifier
light = LGBMClassifier(random_state=1)
light.fit(x_train_scaled, y_train)
light_pred = light.predict(x_test_scaled)


# In[22]:


#classification report
print(classification_report(y_test, light_pred, digits=4))


# In[23]:


#The accuracy on the test
print( accuracy_score(y_test, light_pred))


# In[24]:


#extra tree classifier
from sklearn.ensemble import ExtraTreesClassifier
tree = ExtraTreesClassifier(random_state=1)
tree.fit(x_train_scaled, y_train)
tree_pred = tree.predict(x_test_scaled)
print(classification_report(y_test, tree_pred, digits=3))


# In[25]:


#The accuracy on the test
print( accuracy_score(y_test, tree_pred))


# In[27]:


from sklearn.model_selection import RandomizedSearchCV ,StratifiedKFold



n_estimators = [50, 100, 300, 500, 1000]
min_samples_split = [2, 3, 5, 7, 9]
min_samples_leaf = [1, 2, 4, 6, 8]
max_features = ['auto', 'sqrt', 'log2', None] 
hyperparameter_grid = {'n_estimators': n_estimators,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}
clf = RandomizedSearchCV(tree, hyperparameter_grid, random_state=1)
search = clf.fit(x_train_scaled, y_train)
clf_pred = clf.predict(x_test_scaled)
print(classification_report(y_test, clf_pred, digits=4))


# In[28]:


#checking for the best parameter for the model with RandomSearch
search.best_params_.values()


# In[29]:


#The accuracy on the test
print( accuracy_score(y_test, clf_pred))


# In[30]:


#Hyperparameter tuned ExtraTreesClassifier 

model = ExtraTreesClassifier(n_estimators = 1000 , min_samples_split = 2 , min_samples_leaf = 8, max_features = None)
model.fit(x_train_scaled,y_train)
Xttree_predtund = tree.predict(x_test_scaled)
print( accuracy_score(Xttree_predtund,y_test)) #use inbuilt class feature_importances of tree based classifiers


# In[31]:


#plot graph of feature importances for better visualization
feat_importances = clf.best_estimator_.feature_importances_

sorted (zip(feat_importances,X),reverse = True)
#feat_importances.nlargest().plot(kind='barh');

