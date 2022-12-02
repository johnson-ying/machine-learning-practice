
# implement Gaussian naive bayes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('diabetes.csv')

#test and train data
X = data.to_numpy()
X = X[:,:-1]
Y = data['Outcome'].to_numpy()

# pd.plotting.scatter_matrix(data)

idx = np.random.choice(len(X), int(0.8*len(X)))
testidx = [i for i in range(len(X)) if i not in idx]

Xtrain = X[idx]
Ytrain = Y[idx]

Xtest = X[testidx]
Ytest = Y[testidx]

scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

#some features are not gaussian, but we'll ignore




#Gaussian naive bayes
class MyGaussianNB:
    def __init__(self,):
        pass
    
    def fit(self, Xtrain, Ytrain):
        #derive
        #recall p(y=k|x) = p(x|y=k) * p(k) / p(x) 
        #p(x) is constant for all k, remove it
        #p(y=k|x) = p(x|y=k) * p(y=k)
        #p(y=k|x) = log(p(x|y=k)) + log(p(y=k))
        #p(y=k|x) = log( D∏d 1/np.sqrt(2*np.pi*np.var(x)_dk) * np.exp(-0.5 * (x_d - np.mean(x)_dk)**2 / np.var(x)_dk**2) ) + log(p(y=k))

        self.K = len(np.unique(Ytrain)) #K classes
        self.D = Xtrain.shape[1] # D dimensions
    
        #calculate p(y=k) which is a priori dependent on training data
        self.p_k = np.zeros((self.K))
        for i in range(self.K):
            self.p_k[i] = len(Ytrain[Ytrain==i])/len(Ytrain)
        self.p_k = np.log(self.p_k) #1 x K
           
        #that takes care of log(p(y=k))
        #now, focus on remaining eqn
        #p(y=k|x) = log( D∏d 1/np.sqrt(2*np.pi*np.var(x)_dk) * np.exp(-0.5 * (x_d - np.mean(x)_dk)**2 / np.var(x)_dk**2) )
        #p(y=k|x) = D∑d [log(1/np.sqrt(2*np.pi*np.var(x)_dk)) -0.5 * (x_d - np.mean(x)_dk)**2 / np.var(x)_dk**2]
        #p(y=k|x) = D∑d [log(1/np.sqrt(2*np.pi*np.var(x)_dk))] -  D∑d[0.5 * (x_d - np.mean(x)_dk)**2 / np.var(x)_dk**2]

        #focus on first summation expression  
        self.nk = np.zeros((self.K))
        for i in range(self.K):
            self.nk[i] = np.sum( np.log(1) - np.log( np.sqrt(2 * np.pi * np.var(Xtrain[Ytrain==i], axis = 0)**2) ))
        

        #now, focus on last part of eqn 
        #D∑d[0.5 * (x_d - np.mean(x)_dk)**2 / np.var(x)_dk**2]
        
        #we'll just calculate the means and variance, and we'll calculate the rest in predict function
        self.means = np.zeros((self.D, self.K))  
        self.vars = np.zeros((self.D, self.K))  
        for i in range(self.K):
            self.means[:,i] = np.mean(Xtrain[Ytrain==i], axis = 0)
            self.vars[:,i] = np.var(Xtrain[Ytrain==i], axis = 0)

    def predict(self, X):
        preds = np.zeros((len(X), self.K)) 
        
        #loop through each possible class
        for i in range(self.K):

            xu = 0.5 * (X - self.means[:,i])**2 #N x D

            xddk = xu.dot(self.vars[:,i]**(-2)) #N x 1
            
            #the log of the likelihood term
            likelihood = self.nk[i] - xddk #N x K
            
            #add to the log of the a priori term and store the prediction 
            preds[:,i] = likelihood + self.p_k[i]
        return np.argmax(preds, axis = 1)
    
    def score(self, X, Y):
        pred = self.predict(X)
        return np.mean(pred == Y)


model = MyGaussianNB()

model.fit(Xtrain, Ytrain)
model.predict(Xtrain)
model.score(Xtrain, Ytrain)
model.score(Xtest, Ytest)


#compare to benchmark
from sklearn.naive_bayes import GaussianNB

benchmark = GaussianNB()

benchmark.fit(Xtrain, Ytrain)
benchmark.predict(Xtrain)
benchmark.score(Xtrain, Ytrain)
benchmark.score(Xtest, Ytest)


print("implementation train score: ", model.score(Xtrain, Ytrain))
print("implementation test score: ", model.score(Xtest, Ytest))

print("benchmark train score: ", benchmark.score(Xtrain, Ytrain))
print("benchmark test score: ", benchmark.score(Xtest, Ytest))


#similar but not exact results as sklearn.. where is the inconsistency?
