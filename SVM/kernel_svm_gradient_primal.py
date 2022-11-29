#implement kernel method for linear SVM

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

#Linear SVM with kernel trick - never updating weights
class SVM():
    def __init__(self, kernel, C = 1.0):
        self.kernel = kernel
        self.C = C
        
    def train_objective(self):
        # return 0.5 * self.u.dot(self.K.dot(self.u)) + self.C * np.sum( np.maximum( 0, 1 - self.YuKb))
        return 0.5 * (self.u.dot(self.K)).dot(self.u) + self.C * np.sum( np.maximum( 0, 1 - self.YuKb))
        
    def fit(self, X, Y, lr = 1e-5, n_iter = 400):
        self.Xtrain = X
        self.Ytrain = Y
        self.u = np.random.randn(self.Xtrain.shape[0]) #N x 1
        self.b = 0
        
        #kernel
        self.K = self.kernel(X,X)

        #gradient descent
        self.losses = []
        for _ in range(n_iter):
            
            #important to put this bit in the for loop, otherwise margins would never update
            self.uKb = self.u.dot(self.K) + self.b
            self.YuKb = Y * self.uKb #AKA. the margins
                        
            loss = self.train_objective()
            self.losses.append(loss)
            
            #grad is Ku-C*sum of yK only for data points where the margin is less than 1
            idx = np.where(self.YuKb < 1)[0]

            # grad_u = self.K.dot(self.u) - self.C * Y[idx].dot(self.K[idx])
            grad_u = self.K.dot(self.u) - self.C * self.K[:,idx].dot(Y[idx])
            self.u -= lr * grad_u
            
            grad_b = -self.C * np.sum(Y[idx])
            self.b -= lr * grad_b
        
        plt.plot(self.losses)
        plt.title('losses per iteration')

    def predict(self, X):
        return np.sign( self.u.dot(self.kernel(self.Xtrain, X)) + self.b)
  
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)



model = SVM(rbf)
    
model.fit(X, Y, lr = 1e-4, n_iter = 4000)
    
model.predict(X)
model.score(X,Y)
