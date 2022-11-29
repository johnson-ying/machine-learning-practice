# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:17:12 2022

@author: Johnson
"""

#implement a "fake" random tree using a bagging tree
#for each bagged decision tree, sample a random column of features

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#get mushroom data
data = pd.read_csv('agaricus-lepiota.data')

#target data is first column
# e = edible = 0, p = poisonous = 1
data['p'] = data['p'].apply(lambda row: 0 if row[0]=='e' else 1)
Y = data['p'].to_numpy()

#transform remaining data 
X = data[data.columns[1:]]


#create a label encoder for each column of the data
n = columns = len(X.columns)
labels = {}

c = 0
for col in X.columns:
    encoder = LabelEncoder()
    encoder.fit(data[col])
    labels[c] = encoder
    c += 1

#find dimensionality
dimensionality = 0
for col, enc in labels.items():
    dimensionality += len(enc.classes_)

XX = np.zeros((len(X), dimensionality)) #essentially a one hot version

i = 0
for col, enc in labels.items():
    K = len(enc.classes_)
    headers = X.columns
    XX[range(len(X)), enc.transform(X[headers[col]]) + i] = 1
    i += K


#create train and test
idx = np.random.choice(len(XX), int(0.5*len(XX)), replace=False)
testidx = [i for i in range(len(XX)) if i not in idx]

Xtrain = XX[idx]
Ytrain = Y[idx]

Xtest = XX[testidx]
Ytest = Y[testidx]

K = np.max(Ytrain)+1 #number of classes


#
#
#

# create fake random tree
class FakeRandomTreeRegressor:
    def __init__(self, n_trees):
        self.n_trees = n_trees
    
    def fit(self, X, Y):
        
        Ntrain = int(0.6 * len(X)) # number of bootstrapped samples for each tree
        Dtrain = int(np.sqrt(X.shape[1]))# number of feature columns for each tree
        
        #create all trees
        self.models = []
        for _ in range(self.n_trees):
            m = DecisionTreeClassifier()
            self.models.append(m)
            
        #train
        self.featureidx = []   
        for m in self.models:
            rowidx = np.random.choice(len(X), Ntrain, replace = True)
            colidx = np.random.choice(X.shape[1], Dtrain, replace = True)
            
            self.featureidx.append(colidx) #save the column idx for each tree - necessary for making later predictions
            
            Xtrain = X[rowidx]
            Xtrain = Xtrain[:,colidx]
            Ytrain = Y[rowidx]
        
            m.fit(Xtrain, Ytrain)
        
    def predict(self, X):
        allpreds = np.zeros((len(X), K)) #one-hot
        i = 0
        for m in self.models:
            selectiveX = X[:,self.featureidx[i]] #only specific specific columns that each tree was trained on
            allpreds[range(len(X)), m.predict(selectiveX)] += 1
            i += 1
        return np.argmax(allpreds, axis = 1)
    
    def score(self, X, Y):
        yhat = self.predict(X)
        return np.mean(yhat == Y)
    
    
    
    
model = FakeRandomTreeRegressor(n_trees = 10)
model = FakeRandomTreeRegressor(n_trees = 1000)

model.fit(Xtrain, Ytrain)

model.predict(Xtest)
model.score(Xtest, Ytest)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




