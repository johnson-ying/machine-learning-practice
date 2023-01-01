
#Use PCA to do feature extraction on data before using a gaussian naive bayes model for mnist data
#naive bayes assumption is that all features are independent
#in parallel, transformed PCA data features are all independent
#see whether applying PCA transformation improves naive bayes performance

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

Xtrain = Xtrain.reshape((len(Xtrain), -1)) #reshape to N x D 
Xtest = Xtest.reshape((len(Xtest), -1)) #reshape to N x D 

#normalize data
Xtrain, Xtest = Xtrain/255, Xtest/255


#Gaussian naive bayes
from scipy.stats import multivariate_normal as mvn
class MyGaussianNB:
    def __init__(self,):
        pass
    
    def fit(self, Xtrain, Ytrain):

        self.K = len(np.unique(Ytrain)) #K classes
        self.D = Xtrain.shape[1] # D dimensions
    
        #calculate log priori 
        self.logprior = np.zeros((self.K))
        for i in range(self.K):
            self.logprior[i] = len(Ytrain[Ytrain==i])/len(Ytrain)
        self.logprior = np.log(self.logprior) #1 x K
           
        #calculate log likelihood terms
        self.means = np.zeros((self.K, self.D))  
        self.vars = np.zeros((self.K, self.D)) 
        for i in range(self.K):
            self.means[i] = np.mean(Xtrain[Ytrain==i], axis = 0)
            self.vars[i] = np.var(Xtrain[Ytrain==i], axis = 0) + 1e-4 #add smoothing to variance
        
    def predict(self, X):
        preds = np.zeros((len(X), self.K)) 
        
        for k, pr, m, v in zip(range(self.K), self.logprior, self.means, self.vars):
            preds[:,k] = mvn.logpdf(X, mean=m, cov=v) + pr
        
        return np.argmax(preds, axis = 1)
    
    def score(self, X, Y):
        pred = self.predict(X)
        return np.mean(pred == Y)




model = MyGaussianNB()
model.fit(Xtrain, Ytrain)


from sklearn.decomposition import PCA
pca = PCA(100)
ztrain = pca.fit_transform(Xtrain)
ztest = pca.transform(Xtest)

model2 = MyGaussianNB()
model2.fit(ztrain, Ytrain)

#no PCA transformation
print(model.score(Xtrain, Ytrain))
print(model.score(Xtest, Ytest))

#with PCA transformation
print(model2.score(ztrain, Ytrain))
print(model2.score(ztest, Ytest))
