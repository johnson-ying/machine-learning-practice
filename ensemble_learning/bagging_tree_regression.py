#implement bagging decision trees using premade decision trees from sklearn
#train on bootstrapped data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#create data 
X = np.linspace(1,100,100) 
Y = np.sin(X)

#test out single decision tree performance
model = DecisionTreeRegressor()

#number of bootstrapped samples
N = 30

#bootstrap and create train data
idx = np.random.choice(100, size = N, replace = False).astype(np.int32)
Xtrain = X[idx].reshape(N,1)
Ytrain = Y[idx].reshape(N,1)

model.fit(Xtrain,Ytrain)

Xtest = X.reshape(100,1)
Ytest = Y.reshape(100,1)

yhat = model.predict(Xtest)

#visualize predicted against ground truth
plt.plot(yhat)
plt.plot(Ytest)



## now, create BaggedTreeRegressor and compare performance

class BaggedTreeRegressor:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        
    def fit(self, X, Y):
        N = len(X)//2 #number of bootstrapped samples
        self.models = []
        
        #create multiple models
        for _ in range(self.n_trees):
            m = DecisionTreeRegressor()
            self.models.append(m)
        
        #train
        for m in self.models:
            idx = np.random.choice(100, size = N, replace = True).astype(np.int32)
            Xtrain = X[idx].reshape(N,1)
            Ytrain = Y[idx].reshape(N,1)
            m.fit(Xtrain,Ytrain)
        
    def predict(self,X):
        allyhat = np.empty((len(self.models), len(X)))
        for i in range(len(self.models)):
            yhat = self.models[i].predict(X)
            allyhat[i,:] = yhat
        return np.mean(allyhat,axis = 0)
    
    def score(self, X, Y):
        d1 = Y - self.predict(X).reshape(100,1)
        d2 = Y - Y.mean()
        return 1 - d1.T.dot(d1) / d2.T.dot(d2)[0][0]
        
    

BaggedTree = BaggedTreeRegressor(n_trees=1000)

X = np.linspace(1,100,100).reshape(100,1)
Y = np.sin(X).reshape(100,1)
        
BaggedTree.fit(X, Y)    
        
yhat = BaggedTree.predict(X)
        
#visualize predicted against ground truth
plt.plot(yhat)
plt.plot(Y)     
        
BaggedTree.score(X,Y)[0][0]
