
#N-layer ANN in theano

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


# load MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# standardize data between 0 and 1
test_X = test_X / 255
train_X = train_X / 255

# resize to N x D 
train_X = np.resize(train_X, (60000,784))
test_X = np.resize(test_X, (10000,784))


#Layer class 
class Layer():
    def __init__(self, M1, M2, activation):        
        self.W = theano.shared(np.random.randn(M1,M2)/np.sqrt(M1))
        self.b = theano.shared(np.zeros((M2)))
        self.activation = activation
        self.params = [self.W, self.b]
        
    def forward(self, X):
        return self.activation(X.dot(self.W) + self.b)

#ANN class
class myANN():
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes

    def fit(self, X, Y, lr, n_iter, B, n_batches):

        M1 = X.shape[1]
        K = len(np.unique(Y))

        #create layers - relu
        self.layers = []
        for M2 in self.layer_sizes:
            layer = Layer(M1,M2,T.nnet.relu)
            self.layers.append(layer)
            M1 = M2
        #final layer - softmax
        finallayer = Layer(M1, K, T.nnet.softmax)
        self.layers.append(finallayer)

        #store all layer params
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        #graph
        thX = T.matrix('X')
        thT = T.ivector('targets')
        
        thZ = thX
        for layer in self.layers:
            thZ = layer.forward(thZ)
        thY = thZ
        pred = T.argmax(thY, axis = 1)
        
        #cost
        cost = -T.mean( T.log( thY[T.arange(thT.shape[0]), thT] ) )
        grads = T.grad(cost, self.params)
        
        #updates
        updates = [(p, p - lr * g) for p, g in zip(self.params, grads)]
        
        #train function
        train = theano.function(inputs = [thX, thT], 
                                updates = updates)
        
        self.get_pred = theano.function(inputs = [thX, thT],
                                   outputs= [cost, pred])
        
        #train
        self.costs = []
        for i in range(n_iter):  
            #batch gd
            for j in range(n_batches):

                #batches
                Xbatch = X[j*B:(j+1)*B,:]
                Ybatch = Y[j*B:(j+1)*B]
                
                train(Xbatch, Ybatch)
                
                c, _ = self.get_pred(Xbatch, Ybatch)
                self.costs.append(c)
                
                print('iter: ', i, ' batch: ', j, ' cost: ', c)

        
        plt.plot(self.costs)
                
    def predict(self, X, Y):
        c, p = self.get_pred(X, Y)
        return p
        
    def score(self, X, Y):
        pred = self.predict(X, Y)
        return np.mean(Y == pred)

N = len(train_X)
B = 1024
n_batches = int(N//B)
lr = 1e-1
n_iter = 2

model = myANN([20,10])
model.fit(train_X, train_y, lr, n_iter, B, n_batches)


model.score(test_X, test_y)

model.score(train_X, train_y)
