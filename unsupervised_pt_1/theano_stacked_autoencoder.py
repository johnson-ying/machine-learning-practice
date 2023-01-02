
#Create a stacked autoencoder
#Each layer is the hidden layer of a separate autoencoder
#Each autoencoder was trained separately (greedy layer wise pre-training)
#
#didnt run the code or tune hyperparameters, theano isnt optimized for speed, just made sure the general architecture works

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
Xtrain = Xtrain[0:500,:]
Ytrain = Ytrain[0:500]



#Autoencoder class
class AutoEncoder():
    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2
        self.w = theano.shared(np.random.randn(M1,M2)/np.sqrt(M1))
        self.bh = theano.shared(np.zeros((M2)))
        self.bo = theano.shared(np.zeros((M1)))
        self.params = [self.w, self.bh, self.bo]
        self.f_params = [self.w, self.bh]
        
    def fit(self, X, lr, B, n_batches, n_iter):
        
        thX = T.matrix('input')
        z = thX
        #forward
        hidden = T.nnet.sigmoid(z.dot(self.w) + self.bh)
        out = T.nnet.sigmoid(hidden.dot(self.w.T) + self.bo)
        
        #cost 
        cost = T.mean((thX - out)*(thX - out))
        
        #gradients
        grad = T.grad(cost, self.params)
        
        #param updates
        updates = [(p, p - lr*g) for p, g in zip(self.params, grad)]
        
        #train function
        train = theano.function(inputs = [thX],
                                updates = updates,
                                outputs = cost)
        
        self.get_hidden_layer = theano.function(inputs = [thX], 
                                    outputs = hidden)
        
        #predict the output
        self.pred = theano.function(inputs = [thX], 
                                    outputs = out)
        
        #train loop
        self.costs = []
        for i in range(n_iter):
            for j in range(n_batches):
                Xbatch = X[j*B:(j+1)*B,:]
                
                c = train(Xbatch)
                self.costs.append(c)
                print('iter: ', i, 'batch #: ', j, 'cost: ', c)
                
        plt.plot(self.costs)
    
    #get hidden layer
    def get_hidden(self, X):
        #why didnt we just return self.get_hidden_layer ?
        #well, in the later DeepAutoEncoder class, we need to retrieve the hidden layers on the forward pass
        #and the input would necessarily be a tensor
        #but the self.get_hidden_layer function cannot take a tensor as an input
        #so we need a general forward eqn to pass the tensor
        #see line 161-164. see what would happen if you tried to return self.get_hidden_layer
        #in contrast, line 142 does make use of the self.get_hidden_layer function
        Z = T.nnet.sigmoid(X.dot(self.w) + self.bh)
        return Z
    
    #get recreation
    def predict(self, X):
        return self.pred(X)

#testing autoencoder
# N, D = Xtrain.shape
# B = 64
# n_batches = int(N//B)

# AE = AutoEncoder(D,300)
# AE.fit(Xtrain, lr = .5, B = B, n_batches = n_batches, n_iter = 2)

#Layer class for final layer in deep network
class Layer():
    def __init__(self, M1, M2, activation):
        self.w = theano.shared(np.random.randn(M1,M2)/np.sqrt(M1))
        self.b = theano.shared(np.zeros((M2)))
        self.params = [self.w, self.b]
        self.activation = activation
    
    def forward(self, X):
        return self.activation(X.dot(self.w) + self.b)
   

#deep network, consisting of multiple AEs
class DeepAutoEncoder():
    def __init__(self, ae_sizes):
        self.ae_sizes = ae_sizes 
        
    def fit(self, X, Y, lr, B, n_batches, n_iter):
        
        M1 = X.shape[1] #dimension D
        K = len(set(Y))
        
        #instantiate all single-layer autoencoders
        self.ae = []
        c = 1
        hidden = X
        for M2 in self.ae_sizes:
            ae = AutoEncoder(M1, M2)
            print('training autoencoder #: ', c)
            ae.fit(hidden, lr, B, n_batches, n_iter)
            self.ae.append(ae)
            M1 = M2
            c += 1
            hidden = ae.get_hidden_layer(hidden) #the next ae will use the current hidden as input
            
        #add final prediction layer
        final_layer = Layer(M1, K, T.nnet.softmax)
        
        #add all params
        self.params = []
        for ae in self.ae:
            self.params += ae.f_params
        self.params += final_layer.params
    
    
        thX = T.matrix('X')
        thT = T.ivector('targets')
        
        #forward
        curr_input = thX
        for ae in self.ae:
            curr_input = ae.get_hidden(curr_input)
        out = final_layer.forward(curr_input)
        pred = T.argmax(out, axis = 1)
        
        #cost
        cost = -T.mean( T.log( out[T.arange(thT.shape[0]), thT] ) )

        #gradients
        grad = T.grad(cost, self.params)
        
        #param updates
        updates = [(p, p - lr*g) for p, g in zip(self.params, grad)]
        
        #train function
        train = theano.function(inputs = [thX, thT],
                                updates = updates,
                                outputs = cost)
        
        #predict the output
        self.pred = theano.function(inputs = [thX], 
                                    outputs = pred)
        #train loop
        self.costs = []
        for i in range(n_iter):
            for j in range(n_batches):
                Xbatch = X[j*B:(j+1)*B,:]
                Ybatch = Y[j*B:(j+1)*B]
                   
                c = train(Xbatch, Ybatch)
                self.costs.append(c)
                print('iter: ', i, 'batch #: ', j, 'cost: ', c)
                
        plt.plot(self.costs)
    
    #get recreation
    def predict(self, X):
        return self.pred(X)

    def score(self, X, Y):
        p = self.predict(X)
        return np.mean(p == Y)

N, D = Xtrain.shape
B = 64
n_batches = int(N//B)

model = DeepAutoEncoder([300,100,50])
model.fit(Xtrain, Ytrain, lr = .5, B = B, n_batches = n_batches, n_iter = 1)


model.score(Xtrain, Ytrain)
