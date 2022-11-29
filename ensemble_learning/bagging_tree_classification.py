#similar as bagging_tree_regression.py, but for classification

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier

# load MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# standardize data between 0 and 1
test_X = test_X / 255
train_X = train_X / 255

plt.imshow(test_X[0])

# resize to N x D 
train_X = np.resize(train_X, (60000,784))
test_X = np.resize(test_X, (10000,784))

train_y = np.resize(train_y, (len(train_y),1))
test_y = np.resize(test_y, (len(test_y),1))



#single decision tree model 
model = DecisionTreeClassifier()

model.fit(train_X, train_y)

yhat = model.predict(test_X)
yhat = np.resize(yhat, (len(yhat),1))

model.score(test_X, test_y)



#Bagged decision tree model

(train_X, train_y), (test_X, test_y) = mnist.load_data()
X = np.concatenate((train_X, test_X))
Y = np.concatenate((train_y, test_y))

X = X / 255

X = np.reshape(X, (70000,28*28))
Y = np.reshape(Y, (70000,1))

N = int(0.7 * 70000)
K = np.max(Y)+1 #number of classes

class BaggedTreeClassification:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        
    def fit(self, X, Y):
        
        #create all models
        self.models = []
        for _ in range(self.n_trees):
            m = DecisionTreeClassifier()
            self.models.append(m)
        
        #train each model
        for m in self.models:  
            Xtrain = X[:N]
            Ytrain = Y[:N]
            
            m.fit(Xtrain,Ytrain)
    
    #really inefficient as a first pass solution
    # def predict(self, X):
    #     allpreds = np.empty((self.n_trees, len(X), 1)) #ntrees x N x 1
    #     for i in range(len(self.models)):
    #         yhat = self.models[i].predict(X)
    #         yhat = np.reshape(yhat, (len(X), 1))
    #         allpreds[i,:,:] = yhat
    #     allpreds = np.reshape(allpreds, (self.n_trees, len(X)))
    #     allpreds = allpreds.astype(np.int32)
        
    #     finalpreds = np.empty((len(X)))
    #     for i in range(len(X)):
    #         finalpreds[i] = np.bincount(allpreds[:,i]).argmax()
        
    #     return np.reshape(finalpreds, (len(X), 1))
    
    
    #more elegant solution
    def predict(self, X):
        allpreds = np.zeros((len(X),K)) #one hot
        for m in self.models:
            allpreds[np.arange(len(X)),m.predict(X)] += 1
        return np.argmax(allpreds, axis = 1)

    def score(self, X, Y):
        yhat = self.predict(X)
        yhat = np.reshape(yhat, (len(yhat,),1))
        return np.sum(yhat == Y)/len(Y)
        
        
        
BaggedTrees = BaggedTreeClassification(n_trees = 1)
BaggedTrees = BaggedTreeClassification(n_trees = 10)

BaggedTrees.fit(X,Y)
BaggedTrees.score(X,Y)   
        
