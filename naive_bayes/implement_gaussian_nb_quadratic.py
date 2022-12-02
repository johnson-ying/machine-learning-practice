# same as implement_gaussian_nb.py, but implement Gaussian naive bayes using its quadratic form
# should also generalize to non-naive scenarios where features are correlated

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
        
        self.K = len(np.unique(Ytrain)) #K classes
        self.D = Xtrain.shape[1] # D dimensions
        
        #derive
        #recall p(y=k|x) = p(x|y=k) * p(k) / p(x) 
        #p(x) is constant for all k, remove it
        #p(y=k|x) = p(x|y=k) * p(y=k)
        #p(y=k|x) = log(p(x|y=k)) + log(p(y=k))
        
        #calculate p(y=k) which is a priori dependent on training data
        self.p_k = np.zeros((self.K))
        for i in range(self.K):
            self.p_k[i] = len(Ytrain[Ytrain==i])/len(Ytrain)
        self.p_k = np.log(self.p_k) #1 x K
        
        #log of the likelihood first part 
        self.cov = np.zeros((self.D, self.D, self.K))
        self.log_c = np.zeros((self.K))
        for i in range(self.K):
            self.cov[:,:,i] = np.cov(Xtrain[Ytrain==i].T) #D x D
            self.log_c[i] = np.log(1) - np.log(np.sqrt( (2*np.pi)**self.D * np.linalg.det(self.cov[:,:,i]) ))
        
        #log of likelihood second part will be in predict function
        self.means = np.zeros((self.D, self.K))  
        self.vars = np.zeros((self.D, self.K))  
        for i in range(self.K):
            self.means[:,i] = np.mean(Xtrain[Ytrain==i], axis = 0)
            self.vars[:,i] = np.var(Xtrain[Ytrain==i], axis = 0)
        
    def predict(self, X):
        preds = np.zeros((len(X), self.K))
        
        for k, pr, logc in zip(range(self.K), self.p_k, self.log_c):
            
            c = self.cov[:,:,k]
            
            #quadratic scalars
            A = -0.5 * np.linalg.inv(c)
            B = np.linalg.inv(c).dot(self.means[:,k]) #D x 1
            C = logc - 0.5 * (self.means[:,k].dot(np.linalg.inv(c))).dot(self.means[:,k]) + pr #just a constant
            
            #quadratic terms
            Ax = (X.dot(A)).dot(X.T) #N x N but we want the squared terms - hence the diagonal only
            Ax = np.diag(Ax) #N x 1 
            Bx = X.dot(B) #N x 1
            
            preds[:,k] = Ax + Bx + C #gives  constant
        
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
