# implement svm dual + regularization via projected gradient descent

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# breast cancer data
data = load_breast_cancer()

X = data['data']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# X = (X - np.mean(X)) / np.std(X)

Y = data['target']
#make sure 0 targets are -1
Y[Y==0] = -1

#kernels

#linear
def linear(X1, X2, c = 0):
    return X1.dot(X2.T) + c #size N x N

#gaussian
#K(x,x') = exp ( - ||x-x'||^2 * gamma where gamma = 1/2 sigma^2)
def rbf (X1, X2, gamma = None): 
    if gamma == None:
        gamma = 1.0/X1.shape[-1]
    #3 cases: goal is always to get it into form N1 x N2 
    if np.ndim(X1) == 1 and np.ndim(X2) == 1:
        result = np.exp(-gamma * np.linalg.norm(X1-X2)**2) #1 x 1 
    elif (np.ndim(X1) > 1 and np.ndim(X2) == 1) or (np.ndim(X1) == 1 and np.ndim(X2) > 1):
        result = np.exp(-gamma * np.linalg.norm(X1-X2, axis = 1)**2) #N1 x N2   
    elif np.ndim(X1) > 1 and np.ndim(X2) > 1:
        #clever trick
        result = np.exp(-gamma * np.linalg.norm(X1[:,np.newaxis] - X2[np.newaxis,:], axis = 2)**2) #N1 x N2   
    return result    

#sigmoid
def sigmoid(X1, X2, gamma = 0.05, c = 1):
    return np.tanh(gamma * X1.dot(X2.T) + c)

class SVM():
    def __init__(self, kernel, C = 1.0):
        self.kernel = kernel
        self.C = C
    
    def train_objective(self):
        return np.sum(self.alphas) - 0.5 * np.sum(self.YYK * np.outer(self.alphas, self.alphas))
    
    def fit(self, X, Y, lr = 1e-5, n_iter = 400):
        self.Xtrain = X
        self.Ytrain = Y
        self.alphas = np.random.random(np.shape(X)[0])
        self.b = 0
        
        #kernel matrix 
        self.K = self.kernel(X,X)
        self.YY = np.outer(Y,Y)
        self.YYK = self.K * self.YY
        
        #gradient ascent
        self.losses = []
        for _ in range(n_iter):
            loss = self.train_objective()
            self.losses.append(loss)
            
            #gradient is J = 1 - Ga
            grad_a = np.ones(np.shape(X)[0]) - self.YYK.dot(self.alphas) 
            self.alphas += lr * grad_a
            
            #clamp a between 0 and C (which is 1 by default)
            self.alphas[self.alphas < 0.] = 0.
            self.alphas[self.alphas > self.C] = self.C
            
        #calculate b
        #b = y - wx
        #and recall that w = a*y*x
        #which would give you Y - alphas * y * kernel matrix of x
        idx = np.where((self.alphas) > 0 & (self.alphas < self.C))[0]
        allb = Y[idx] - (self.alphas * Y).dot(self.kernel(X,X[idx]))#final size is 1 x N2
        self.b = np.mean(allb)
        
        plt.plot(self.losses)
        plt.title('loss per iteration')
    
    def predict(self,X):
        return np.sign( (self.alphas * self.Ytrain).dot(self.kernel(self.Xtrain,X) + self.b) ) 
        
    def score(self,X,Y):
        P = self.predict(X)
        return np.mean(Y==P)
        

model = SVM(rbf)
    
model.fit(X, Y, lr = 1e-3, n_iter = 400)
    
model.predict(X)
model.score(X,Y)
