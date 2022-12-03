
#implement bernoulli naive bayes
#follows same format as gaussian implementation
#but this time around, it'll be less cluttered with derivation comments

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('dna.csv')

#test and train data
X = data.to_numpy()
X = X[:,:-1]
Y = data['class'].to_numpy()
Y -= 1


idx = np.random.choice(len(X), int(0.8*len(X)))
testidx = [i for i in range(len(X)) if i not in idx]

Xtrain = X[idx]
Ytrain = Y[idx]

Xtest = X[testidx]
Ytest = Y[testidx]


#Bernoulli NB
class myBernoulliNB:
    def __init__(self):
        pass
    
    def fit(self, Xtrain, Ytrain):
        
        self.K = len(np.unique(Ytrain)) #K classes
        D = Xtrain.shape[1] # D dimensions
        
        #log of priors
        self.p_k = np.zeros((self.K))
        for i in range(self.K):
            self.p_k[i] = len(Ytrain[Ytrain==i])/len(Ytrain)
        self.p_k = np.log(self.p_k) #1 x K
        
        #log of likelihood
        #
        #consider a bernoulli distribution
        #p(x) = pi**x * (1-pi)**(1-x)
        #calculate pi: probabilities of 0 or 1 for each D and each K
        #pi = 1, 1-pi = 0
        self.pi = np.zeros((D, self.K))
        for i in range(self.K):
            X_k = Xtrain[Ytrain == i]
            self.pi[:,i] = np.count_nonzero(X_k, axis = 0) / len(X_k) 
        

        #log(p(x|y=k)) = D∑d ( x_d * log(pi_dk) + (1 - x_d) * log(1-pi_dk))
        #can reduce to a linear form Wx + b but this is sufficient to implement
              
    def predict(self, X):
        pred = np.zeros((len(X), self.K))
        for k, p_k in zip(range(self.K), self.p_k):
            
            #using np.nan_to_num and np.isneginf to deal with nan or -inf values
            #eh.. still numerical instability
            log_pi_dk = np.nan_to_num(np.log(self.pi[:,k]))
            log_pi_dk = np.where( np.isneginf(log_pi_dk), 0, log_pi_dk)
            firstterm = X.dot(log_pi_dk)
            
            log_1_minus_pi_dk = np.nan_to_num (np.log( 1 - self.pi[:,k]))
            log_1_minus_pi_dk = np.where( np.isneginf(log_1_minus_pi_dk), 0, log_1_minus_pi_dk)
            secondterm = (1-X).dot(log_1_minus_pi_dk)
        
            loglikelihood = firstterm + secondterm
            pred[:,k] = loglikelihood + p_k
            
        return np.argmax(pred, axis = 1)
        
    def score(self, X, Y):
        pred = self.predict(X)
        return np.mean(pred == Y)
    
    
    
    
model = myBernoulliNB()

model.fit(Xtrain, Ytrain)
model.predict(Xtrain)
model.score(Xtrain, Ytrain)
model.score(Xtest, Ytest)
    
    
    
#compare to benchmark 
from sklearn.naive_bayes import BernoulliNB

benchmark = BernoulliNB()

benchmark.fit(Xtrain, Ytrain)
benchmark.predict(Xtrain)
benchmark.score(Xtrain, Ytrain)
benchmark.score(Xtest, Ytest)


print("implementation train score: ", model.score(Xtrain, Ytrain))
print("implementation test score: ", model.score(Xtest, Ytest))

print("benchmark train score: ", benchmark.score(Xtrain, Ytrain))
print("benchmark test score: ", benchmark.score(Xtest, Ytest))
    
