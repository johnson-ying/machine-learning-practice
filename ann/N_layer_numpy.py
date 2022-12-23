
#N-layer ANN using numpy, and delta recursion to calculate gradients
#assume sigmoid activation in each layer for simplicity 


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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims=True)

def onehot(y,k):
    out = np.zeros((y.shape[0], k))
    for i in range(y.shape[0]):
        out[i,y[i]] = 1
    return out



class Layer():
    def __init__(self, M1, M2):        
        self.W = np.random.randn(M1,M2)
        self.b = np.zeros((M2))
    def forward(self, X):
        self.hidden = sigmoid(X.dot(self.W) + self.b)
        return self.hidden
    
class FinalLayer():
    def __init__(self, M1, M2):        
        self.W = np.random.randn(M1,M2)
        self.b = np.zeros((M2))
    def forward(self, X):
        return softmax(X.dot(self.W) + self.b)

class myANN():
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
    
    def fit(self, X, Y, lr = 1e-4, n_iter = 10):
        
        M1 = X.shape[1]
        K = len(np.unique(Y))
        T = onehot(Y,K)

        #create layers
        self.layers = []
        for M2 in self.layer_sizes:
            layer = Layer(M1,M2)
            self.layers.append(layer)
            M1 = M2
        #final layer
        finallayer = FinalLayer(M1, K)
        self.layers.append(finallayer)
        
        #train
        self.costs = []
        for i in range(n_iter):
            
            print('iter: ', i)
            
            #forward
            hidden = X
            for layer in self.layers:
                hidden = layer.forward(hidden)
            pred = hidden
            
            #cost 
            cost = -1 * np.sum(T * np.log(pred))
            self.costs.append(cost)
            
            #create deltas
            #iterate through layers in reverse
            self.deltas = []
            for j in range(len(self.layers)-1, -1, -1):
                #first delta
                if j == len(self.layers)-1:
                    self.deltas.append(T-pred)
                else:
                    next_w = self.layers[j+1].W
                    curr_z = self.layers[j].hidden
                    d = self.deltas[-1].dot(next_w.T) * curr_z * (1 - curr_z)
                    self.deltas.append(d)
            #reverse order
            self.deltas.reverse()
                    
            #calculate grads using delta
            self.updates_w = []
            self.updates_b = []
            for j in range(len(self.layers)-1, 0, -1):
                layer = self.layers[j]
                prevlayerhidden = self.layers[j-1].hidden
                self.updates_w.append(prevlayerhidden.T.dot(self.deltas[j]))
                self.updates_b.append( np.sum(self.deltas[j], axis = 0))
                
            #and calculate for very first set of weights
            layer = self.layers[0]
            prevlayerhidden = X
            self.updates_w.append(prevlayerhidden.T.dot(self.deltas[0]))
            self.updates_b.append(np.sum(self.deltas[0], axis = 0))
            
            #reverse order
            self.updates_w.reverse()
            self.updates_b.reverse()
            
            #updates
            for n in range(len(self.layers)):
                self.layers[n].W += lr * self.updates_w[n]
                self.layers[n].b += lr * self.updates_b[n]
            
            plt.plot(self.costs)
        
    def predict(self, X):
        hidden = X
        for layer in self.layers:
            hidden = layer.forward(hidden)
        pred = hidden
        return np.argmax(pred, axis = 1)
        
    def score(self, X, Y):
        pred = self.predict(X)
        return np.mean(pred == Y)

model = myANN([50]) #1 layer performs better using sigmoid activation
model.fit(train_X, train_y, lr = 1e-4, n_iter = 50)

model = myANN([50,50,50,50])
model.fit(train_X, train_y, lr = 1e-5, n_iter = 50)

model.score(test_X, test_y)
model.score(train_X, train_y)
