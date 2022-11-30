#attempt 1 at implementing AdaBoost for binary classification
#use decision trees from sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# breast cancer data
data = load_breast_cancer()

X = data['data']
scaler = StandardScaler()
X = scaler.fit_transform(X)

Y = data['target']
#make sure 0 targets are -1
Y[Y==0] = -1

n = int(0.6 * len(X))
idx = np.random.choice(len(X), n)
testidx = [i for i in range(len(X)) if i not in idx]

Xtrain = X[idx]
Ytrain = Y[idx]

Xtest = X[testidx]
Ytest = Y[testidx]


#AdaBoost
class AdaBoost:
    def __init__(self, n_boosts = 10):
        self.n_boosts = n_boosts
        
    def fit (self, X, Y):
        
        # for each model, calculate error
        # update alpha, weights
        # save alpha 
        
        N = X.shape[0]
        
        #create weights for each sample
        #equal, uniform weights to start off with
        self.w = np.ones((N)) * 1/N 

        #create list to store alphas for each model
        self.alphas = []

        #create all models
        self.models = []
        for _ in range(self.n_boosts):
            m = DecisionTreeClassifier(max_depth=1)
            self.models.append(m)
        
        #train 
        for m in self.models:
            
            #fit and get predictions
            m.fit(X, Y, sample_weight = self.w)
            p = m.predict(X)
            p[p==0] = -1
            
            #calculate error
            #recall error = sum of all weights times real vs. pred targets
            #weights for misclassified samples will yield greater error
            self.e = np.sum(self.w * (p != Y))
            #more elegantly, can be written as the dot product 
            
            #calculate a
            #recall a = 1/2 ln (#weighted correct / #weighted incorrect)
            a = 0.5 * (np.log(1-self.e) - np.log(self.e))
            
            #update weights
            #recall w_i = 1 * exp (-y_i * yhat_m-1(xi) )
            self.w *= np.exp(-a * Y * p.flatten())
                        
            #normalize weights
            self.w /= self.w.sum()
            
            #store alpha
            self.alphas.append(a)

    def predict(self, X):
        yhat = np.zeros((len(X)))
        for alpha, model in zip(self.alphas, self.models):
            yhat += alpha * model.predict(X)
        return np.sign(yhat)
    
    def score(self, X, Y):
        pred = self.predict(X)
        return np.mean(pred == Y)
        
    #returns the model score for each individual tree
    def indiv_model_scores(self, X, Y):
        self.accuracy = []
        for alpha, model in zip(self.alphas, self.models):
            yhat = model.predict(X)
            ac = np.mean(yhat == Y)
            self.accuracy.append(ac)
        return self.accuracy
        
    
    
    
model = AdaBoost(n_boosts = 50)

model.fit(Xtrain,Ytrain)

yhat = model.predict(Xtest)
model.score(Xtest,Ytest)

#visualize indiv model accuracies
accuracies = model.indiv_model_scores(Xtest, Ytest)
plt.plot(accuracies)

#visualize weights
plt.plot(model.w)

#visualize alphas
plt.plot(model.alphas)


#question: if weights theoretically get better for later trees, 
# shouldnt alpha weights be higher for later trees instead of first?
