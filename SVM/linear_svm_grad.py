#implement a linear svm using gradient descent

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

class LinearSVM:
    def __init__ (self, C = 2):
        self.C = C
        
    def gradient_descent (self, X, Y, lr = 1e-3, n_iter = 1000):
        N, D = np.shape(X)
        
        #weights and bias
        self.w = np.random.rand(D) 
        self.b = 0
        
        #train
        self.losses = []
        for _ in range(n_iter):
            loss = self.objective(X,Y)
            self.losses.append(loss)
            
            #find misclassified points only
            margins = Y * (X.dot(self.w) + self.b)
            idx = np.where(margins < 1)[0]
            
            #calculate gradients
            grad_w = self.w - self.C * (Y[idx].dot(X[idx])) #make sure dimensions make sense
            grad_b = -self.C * Y[idx].sum()
            
            #update
            self.w -= lr * grad_w
            self.b -= lr * grad_b
        
        plt.plot(self.losses)
        plt.title("loss per iteration")
        
    #objective function
    def objective (self, X, Y): 
        return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1- Y * (X.dot(self.w) + self.b)).sum()
        
    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)
  
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)
    
model = LinearSVM()
    
model.gradient_descent(X, Y)
    
model.predict(X)
model.score(X,Y)
