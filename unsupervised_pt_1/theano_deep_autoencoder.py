
#multi-layer autoencoder where middle layer is dimension 2 to visualize data

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# load MNIST
(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

# standardize data between 0 and 1
Xtrain = Xtrain / 255
Xtest = Xtest / 255

# resize to N x D 
Xtrain = np.resize(Xtrain, (60000,784))
Xtest = np.resize(Xtest, (10000,784))

#my theano isn't optimized, so just test on small subset of data and make sure general architecture works
Xtrain = Xtrain[0:10000,:]
Ytrain = Ytrain[0:10000]

class Layer():
    def __init__(self, M1, M2, activation):
        self.w = theano.shared(np.random.randn(M1,M2)/np.sqrt(M1))
        self.bi = theano.shared(np.zeros((M2)))
        self.bo = theano.shared(np.zeros((M1)))
        self.params = [self.w, self.bi, self.bo]
        self.activation = activation
    
    def forward(self, X):
        return self.activation(X.dot(self.w) + self.bi)
    
    def backward(self, X):
        return self.activation(X.dot(self.w.T) + self.bo)

#deep autoencoder with momentum
class DeepAutoEncoder():
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes

    def fit(self, X, lr = 1e-3, mu = 0.9, batch_sz = 100, n_iter = 30):
        
        N, D = X.shape
        n_batches = int(N//batch_sz)
        
        #create layers
        self.layers = []
        M1 = D
        for M2 in self.layer_sizes:
            layer = Layer(M1, M2, T.nnet.sigmoid)
            self.layers.append(layer)
            M1 = M2
        
        #store params
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
        #momentum params
        self.dparams = [theano.shared(p.get_value() * 0) for p in self.params]
        
        thX = T.matrix('input')
        #forward pass
        z = thX
        for layer in self.layers:
            z = layer.forward(z)
        middle = z
        
        #reverse pass - going in backwards direction
        for n in range(len(self.layers)-1, -1, -1):
            layer = self.layers[n]
            z = layer.backward(z)
        out = z
            
        #cost: either cross entropy or MSE
        # cost = T.mean( (thX - out) * (thX - out))
        cost = -T.mean( thX * T.log(out) + (1-thX) * T.log(1 - out) )
        
        #gradients
        grads = T.grad(cost, self.params)
        
        #updates
        updates = [(p, p + mu * dp - lr * g) for p, dp, g in zip(self.params, self.dparams, grads)
                   ] + [(dp, mu * dp - lr * g) for dp, g in zip(self.dparams, grads)
                        ]
        
        #train function
        train = theano.function(inputs = [thX],
                                updates = updates,
                                outputs = cost)
        
        #predict the output
        self.pred = theano.function(inputs = [thX], 
                                    outputs = out)
        
        #predict the output
        self.latent = theano.function(inputs = [thX], 
                                    outputs = middle)
        
        #train loop
        self.costs = []
        for i in range(n_iter):
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j+1)*batch_sz,:]
                   
                c = train(Xbatch)
                self.costs.append(c)
                print('iter: ', i, 'batch #: ', j, 'cost: ', c)
                
        plt.plot(self.costs)
        
    def get_latent(self, X):
        return self.latent(X)


#theano isnt optimized, so didnt fine tune code or hyperparameters, just made sure structure works
model = DeepAutoEncoder([300,200,2])
model.fit(Xtrain, lr = 0.5, mu = 0.95, batch_sz = 100, n_iter = 3)

#visualize latent representation
latent = model.get_latent(Xtrain)

plt.scatter(latent[:,0], latent[:,1], c = Ytrain, s = 20)
plt.jet()
