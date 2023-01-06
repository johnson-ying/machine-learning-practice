
#generative sampling mnist data using a bayes model
#in classification, we wanted: p(y|x) = p(x|y) * p(y)
#in generative modeling, we want: p(x,y) = p(x|y) * p(y)

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from scipy.stats import multivariate_normal 

(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

Xtrain = Xtrain / 255
Xtest = Xtest / 255

Xtrain = Xtrain.reshape((len(Xtrain), -1))
Xtest = Xtest.reshape((len(Xtest), -1))

#bayes sampler - treat each dimension as an individual gaussian 
class myBayes():
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        
        #calculate priors 
        self.pY = []
        self.K = len(set(Y))
        total = len(Y)
        for k in range(self.K):
            count = len(np.where(Ytrain == k)[0])
            self.pY.append(count/total)
        
        #treat p(x|y) as a multivariate gaussian 
        self.pX_y = []
        for k in range(self.K):
            x_y = X[np.where(Ytrain == k)[0], :]
            mean = np.mean(x_y, axis = 0) #1 x D
            var = np.cov(x_y.T) #D x D 
            self.pX_y.append((mean, var))
        
    #sample given a class y
    def sampleXfromY(self, y):
        return multivariate_normal.rvs(mean = self.pX_y[y][0], cov = self.pX_y[y][1])
    
    #sample randomly
    def sampleX(self):
        y = np.random.choice(self.K, p = self.pY)
        return self.sampleXfromY(y), y
        
            
            
model = myBayes()
model.fit(Xtrain, Ytrain)

#sample from a given class and observe ground truth avg
plt.subplot(121)
xx = model.sampleXfromY(3)
xx = xx.reshape((28,28))
plt.imshow(xx)
plt.gray()

plt.subplot(122)
avg = np.mean( Xtrain[np.where(Ytrain == 3)[0], :], axis = 0 )
avg = avg.reshape((28,28))
plt.imshow(avg)
plt.gray()


#sample randomly
plt.subplot(121)
xx, y = model.sampleX()
xx = xx.reshape((28,28))
plt.imshow(xx)
plt.gray()

plt.subplot(122)
avg = np.mean( Xtrain[np.where(Ytrain == y)[0], :], axis = 0 )
avg = avg.reshape((28,28))
plt.imshow(avg)
plt.gray()
