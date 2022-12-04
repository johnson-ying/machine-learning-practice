
#implement multinomial NB

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import scipy

data = pd.read_csv('bbc_text_cls.csv')

inputs = data['text']
labels = data['labels']

vectorizer = CountVectorizer()

inputs_train, inputs_test, Ytrain, Ytest = train_test_split(inputs, labels, random_state=123)

#count vectorize inputs
Xtrain = vectorizer.fit_transform(inputs_train)
Xtest = vectorizer.transform(inputs_test)

#convert sparse matrix to numpy
Xtrain = scipy.sparse.csr_matrix.toarray(Xtrain)
Xtest = scipy.sparse.csr_matrix.toarray(Xtest)

#labels
vectorizerTest = CountVectorizer()
Ytrain = vectorizerTest.fit_transform(Ytrain)
Ytrain = np.argmax(Ytrain, axis = 1)
Ytrain = np.squeeze(np.asarray(Ytrain))

Ytest = vectorizerTest.transform(Ytest)
Ytest = np.argmax(Ytest, axis = 1)
Ytest = np.squeeze(np.asarray(Ytest))


#Multinomial NB
class myMultinomialNB:
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
        #is reduced to: x_d1 * log(pi_d1_k) + ... + x_d * log(pi_d_k)
        #rewrite in linear form: X.dot(w) where w = vector of log(pi_k)
        #where pi_d_k is probability of seeing x_d in class k
        
        self.pi = np.zeros((D,self.K))
        smoothing = 1e-7 #add some smoothing since we're taking logs on very small numbers
        for i in range(self.K):
            X_k = Xtrain[Ytrain ==i]
            self.pi[:,i] = np.sum(X_k, axis = 0) / np.sum(np.sum(X_k, axis = 0)) + smoothing
        self.pi = np.log(self.pi)
              
    def predict(self, X):
        pred = X.dot(self.pi) + self.p_k
        return np.argmax(pred, axis = 1)
        
    def score(self, X, Y):
        pred = self.predict(X)
        return np.mean(pred == Y)





    
model = myMultinomialNB()

model.fit(Xtrain, Ytrain)
model.predict(Xtrain)
model.score(Xtrain, Ytrain)
model.score(Xtest, Ytest)
    
    
    
#compare to benchmark 
from sklearn.naive_bayes import MultinomialNB

benchmark = MultinomialNB()

benchmark.fit(Xtrain, Ytrain)
benchmark.predict(Xtrain)
benchmark.score(Xtrain, Ytrain)
benchmark.score(Xtest, Ytest)


print("implementation train score: ", model.score(Xtrain, Ytrain))
print("implementation test score: ", model.score(Xtest, Ytest))

print("benchmark train score: ", benchmark.score(Xtrain, Ytrain))
print("benchmark test score: ", benchmark.score(Xtest, Ytest))
    
