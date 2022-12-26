
#N-layer ANN in theano with batch norm

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano.tensor.nnet.bn import batch_normalization_train, batch_normalization_test
from keras.datasets import mnist

# load MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# standardize data between 0 and 1
test_X = test_X / 255
train_X = train_X / 255

# resize to N x D 
train_X = np.resize(train_X, (60000,784))
test_X = np.resize(test_X, (10000,784))


#Layer class - with batch norm 
class Layer():
    def __init__(self, M1, M2, activation): 
        
        #weights - no more bias term 
        self.W = theano.shared(np.random.randn(M1,M2)/np.sqrt(M1))
        self.activation = activation
        
        #gamma and beta to unstandardize during training
        self.gamma = theano.shared(np.ones(M2))
        self.beta = theano.shared(np.zeros(M2))
        
        #batch norm running mean and var
        self.rn_mean = theano.shared(np.zeros(M2))
        self.rn_var = theano.shared(np.zeros(M2))
        
        #store all params - store the running mean and var separately
        self.params = [self.W, self.gamma, self.beta]
        self.rn_params = [self.rn_mean, self.rn_var]
        
    def forward(self, X, is_training):
        a = X.dot(self.W)
        if is_training:
            out, batch_mean, batch_invstd, new_rn_mean, new_rn_var = batch_normalization_train(inputs = a, 
                                                                                               gamma = self.gamma, 
                                                                                               beta = self.beta, 
                                                                                               running_mean = self.rn_mean, 
                                                                                               running_var = self.rn_var)
            return self.activation(out), batch_mean, batch_invstd, new_rn_mean, new_rn_var
        else:
            out = batch_normalization_test(inputs = a, 
                                           gamma = self.gamma, 
                                           beta = self.beta, 
                                           mean = self.rn_mean, 
                                           var = self.rn_var)
            return self.activation(out)

class NormalLayer():
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

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, lr, n_iter, B, n_batches):

        M1 = Xtrain.shape[1]
        K = len(np.unique(Ytrain))

        #create layers - relu
        self.layers = []
        for M2 in self.layer_sizes:
            layer = Layer(M1, M2, T.nnet.relu)
            self.layers.append(layer)
            M1 = M2
        #final layer - softmax - dont do batch norm here
        finallayer = NormalLayer(M1, K, T.nnet.softmax)
        self.layers.append(finallayer)

        #store all layer params
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
        self.rn_params = []
        for layer in self.layers[:-1]:
            self.rn_params += layer.rn_params
            
        #graph
        thX = T.matrix('X')
        thT = T.ivector('targets')
        
        #training 
        new_rn = [] #store running mean and var 
        thZ = thX
        for layer in self.layers[:-1]:
            thZ, batch_mean, batch_invstd, new_rn_mean, new_rn_var = layer.forward(thZ, is_training = 1)
            new_rn.append(new_rn_mean)
            new_rn.append(new_rn_var)
        thZ = self.layers[-1].forward(thZ)
        thY = thZ
        
        #train cost
        cost = -T.mean( T.log( thY[T.arange(thT.shape[0]), thT] ) )
        grads = T.grad(cost, self.params)
        
        #updates
        updates = [(p, p - lr * g) for p, g in zip(self.params, grads)]
        rn_updates = [(p, new_p) for p, new_p in zip(self.rn_params, new_rn)]
        
        updates += rn_updates 
        
        #train function
        train = theano.function(inputs = [thX, thT], 
                                updates = updates)
        
        #get train cost
        get_cost = theano.function(inputs = [thX, thT],
                                   outputs= [cost])
        

        #making predictions 
        pred_Z = thX
        for layer in self.layers[:-1]:
            pred_Z = layer.forward(pred_Z, is_training = 0)
        pred_Z = self.layers[-1].forward(pred_Z)
        pred_Y = pred_Z
        pred = T.argmax(pred_Y, axis = 1)
        
        #prediction cost
        pred_cost = -T.mean( T.log( pred_Y[T.arange(thT.shape[0]), thT] ) )
        
        self.get_pred = theano.function(inputs = [thX, thT],
                                   outputs= [pred_cost, pred])
        
        #train
        self.costs_train = []
        self.costs_test = []
        for i in range(n_iter):  
            #batch gd
            for j in range(n_batches):

                #batches
                Xbatch = Xtrain[j*B:(j+1)*B,:]
                Ybatch = Ytrain[j*B:(j+1)*B]
                
                train(Xbatch, Ybatch)
                
                if j % 10 == 0:
                    cost_train = get_cost(Xbatch, Ybatch)
                    cost_test, _ = self.get_pred(Xtest, Ytest)
                    self.costs_train.append(cost_train)
                    self.costs_test.append(cost_test)
                
                    print('iter: ', i, ' batch: ', j, ' train cost: ', cost_train, ' test cost: ', cost_test)
        
        plt.plot(self.costs_train)
        plt.plot(self.costs_test)
        
    def predict(self, X, Y):
        c, p = self.get_pred(X, Y)
        return p
        
    def score(self, X, Y):
        pred = self.predict(X, Y)
        return np.mean(Y == pred)

N = len(train_X)
B = 512
n_batches = int(N//B)
lr = 0.5
n_iter = 1

model = myANN([30,20])
model.fit(train_X, train_y, test_X, test_y, lr, n_iter, B, n_batches)

#trains fast after ~40 batches on first iteration
model.score(test_X, test_y)

model.score(train_X, train_y)
