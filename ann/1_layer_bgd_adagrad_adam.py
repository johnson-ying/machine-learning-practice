
#1-hidden layer ANN using numpy
#relu activation in middle layer
#implement adam

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


# def sigmoid(x):
#     return 1/(1+np.exp(-x))

def relu(x):
    x[x<0] = 0
    return x

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims=True)

def onehot(y,k):
    out = np.zeros((y.shape[0], k))
    for i in range(y.shape[0]):
        out[i,y[i]] = 1
    return out





#parameters
N, D = train_X.shape
M = 50
K = len(np.unique(train_y))
lr = 1e-5

n_iter = 50
B = 256
n_batches = int(N//B)

#first moment
beta1 = 0.9
mw = 0
mb = 0
mv = 0
mc = 0
 
#second moment
beta2 = 0.99
vw = 0
vb = 0
vv = 0
vc = 0

epsilon = 1e-9
t = 1

W = np.random.randn(D,M)/np.sqrt(M)
b = np.zeros((M))
V = np.random.randn(M,K)/np.sqrt(K)
c = np.zeros((K))

#train
XX = train_X
TT = onehot(train_y, K)
costs = []

for i in range(n_iter):  
    if i%10==0:
        print('iter: ', i)
    #batch gd
    for j in range(n_batches):
        #batches
        X = XX[j*B:(j+1)*B,:]
        T = TT[j*B:(j+1)*B,:]
    
        #forward 
        hidden = relu(X.dot(W) + b)
        y = softmax(hidden.dot(V) + c)
        
        #cost
        cost = np.sum(T * np.log(y)) / B
        costs.append(-1*cost)
        
        #grads - sigmoid
        # grad_V = hidden.T.dot(T - y) #M x K
        # grad_c = np.sum(T - y, axis = 0) #K
        # grad_W = X.T.dot( (T-y).dot(V.T) * hidden * (1-hidden) ) #D x M
        # grad_b = np.sum((T-y).dot(V.T) * hidden * (1-hidden), axis = 0) #M
        
        #grads - relu
        grad_V = hidden.T.dot(T - y) #M x K
        grad_c = np.sum(T - y, axis = 0) #K
        grad_W = X.T.dot( (T-y).dot(V.T) * (hidden > 0) ) #D x M
        grad_b = np.sum((T-y).dot(V.T) * (hidden > 0), axis = 0) #M
        
        #updates - without adam
        # V += lr * grad_V
        # c += lr * grad_c
        # W += lr * grad_W
        # b += lr * grad_b
        
        #adam updates, new mean and variance
        mv = beta1 * mv + (1-beta1) * grad_V
        mc = beta1 * mc + (1-beta1) * grad_c
        mw = beta1 * mw + (1-beta1) * grad_W
        mb = beta1 * mb + (1-beta1) * grad_b
        
        vv = beta2 * vv + (1-beta2) * grad_V * grad_V
        vc = beta2 * vc + (1-beta2) * grad_c * grad_c
        vw = beta2 * vw + (1-beta2) * grad_W * grad_W
        vb = beta2 * vb + (1-beta2) * grad_b * grad_b
        
        #bias correction
        mv_hat = mv / (1 - beta1 ** t)
        mc_hat = mc / (1 - beta1 ** t)
        mw_hat = mw / (1 - beta1 ** t)
        mb_hat = mb / (1 - beta1 ** t)
        
        vv_hat = vv / (1 - beta2 ** t)
        vc_hat = vc / (1 - beta2 ** t)
        vw_hat = vw / (1 - beta2 ** t)
        vb_hat = vb / (1 - beta2 ** t)
        
        #update params
        V += lr * mv_hat / np.sqrt(vv_hat + epsilon)
        c += lr * mc_hat / np.sqrt(vc_hat + epsilon)
        W += lr * mw_hat / np.sqrt(vw_hat + epsilon)
        b += lr * mb_hat / np.sqrt(vb_hat + epsilon)
    
        t += 1
    
plt.plot(costs)


#calculate correct pred rate on test data
z = relu(test_X.dot(W) + b)
pred = softmax(z.dot(V) + c)
pred = np.argmax(pred, axis = 1)

#accuracy rate
print(np.mean(test_y == pred))
