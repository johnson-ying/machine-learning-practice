
#1-hidden layer ANN using numpy
#sigmoid activation in middle layer

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# load MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# standardize data between 0 and 1
test_X = test_X / 255
train_X = train_X / 255

# resize to N x D 
train_X = np.resize(train_X, (60000,784))
test_X = np.resize(test_X, (10000,784))




#parameters
D = train_X.shape[1]
M = 100
K = len(np.unique(train_y))
lr = 1e-4
n_iter = 50

#weights, biases
W = np.random.randn(D,M)
b = np.zeros((M))
V = np.random.randn(M,K)
c = np.zeros((K))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims=True)

def onehot(y,k):
    out = np.zeros((y.shape[0], k))
    for i in range(y.shape[0]):
        out[i,y[i]] = 1
    return out

#train
X = train_X
T = onehot(train_y, K)
costs = []
for i in range(n_iter):
    
    print('iter: ', i)
    
    #forward 
    hidden = sigmoid(X.dot(W) + b)
    y = softmax(hidden.dot(V) + c)
    
    #cost
    cost = np.sum(np.sum(T * np.log(y), axis = 1))
    costs.append(-1*cost)
    
    #grads
    grad_V = hidden.T.dot(T - y) #M x K
    grad_c = np.sum(T - y, axis = 0) #K
    grad_W = X.T.dot( (T-y).dot(V.T) * hidden * (1-hidden) ) #D x M
    grad_b = np.sum((T-y).dot(V.T) * hidden * (1-hidden), axis = 0) #M
    
    #updates - gradient ascent here
    V += lr * grad_V
    c += lr * grad_c
    W += lr * grad_W
    b += lr * grad_b
    
plt.plot(costs)


#calculate correct pred rate on test data
z = sigmoid(test_X.dot(W) + b)
pred = softmax(z.dot(V) + c)
pred = np.argmax(pred, axis = 1)

#accuracy rate ~85%
print(np.mean(test_y == pred))
