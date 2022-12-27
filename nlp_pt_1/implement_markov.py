
# implement a text classifier on poems from edgar allen poe or robert frost
#data from:
#https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/edgar_allan_poe.txt
#https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt
#
# build an individual markov model for each author
# find state probability transition matrix and initial state transition matrix for each
# create a class myBayes which will use this info to determine which class a test sentence belongs to

import numpy as np
import matplotlib.pyplot as plt
import string

#load data
edgar = open('edgar_allan_poe.txt')
data1 = edgar.readlines()
data1 = [item for item in data1 if len(item)>4] #remove lines that are less than length 4

robert = open('robert_frost.txt')
data2 = robert.readlines()
data2 = [item for item in data2 if len(item)>4] #remove lines that are less than length 4

#concatenate data and create labels 
# 0 = edgar 1 = robert
X = data1 + data2
Y = np.concatenate((np.zeros((len(data1))), np.zeros(len(data2))+1 ), axis = 0).astype(np.int32)

idx = np.random.choice(len(X), int(0.8*len(X)))
testidx = [i for i in range(len(X)) if i not in idx]

Xtrain = [X[i] for i in idx]
Ytrain = Y[idx]
Xtest = [X[i] for i in testidx]
Ytest = Y[testidx]

#create dict mapping of unique train words to unique integer
word2idx = {}
counter = 0
for i in Xtrain:
    s = i.lower() #lowercase everything
    s = s.translate(str.maketrans('', '', string.punctuation)) #remove punctuations
    words = s.split() #get indiv words
    
    #store in dict
    for w in words:
        if w not in word2idx.keys():
            word2idx[w] = counter
            counter += 1
#add a final value for unknown words that show up in test but not in train set
word2idx['UnW'] = counter 



#convert all sentences into lists of integers
Xtrain2 = []

for i in Xtrain:
    s = i.lower() #lowercase everything
    s = s.translate(str.maketrans('', '', string.punctuation)) #remove punctuations
    words = s.split() #get indiv words
    
    sen = []
    for w in words:
        sen.append(word2idx[w])
    Xtrain2.append(sen)

Xtrain_edgar2 = [Xtrain2[i] for i in np.where(Ytrain==0)[0]]
Xtrain_robert2 = [Xtrain2[i] for i in np.where(Ytrain==1)[0]]



# Train a markov model: compute initial state probability (pi) and state transition probabilities (A)
pi_edgar = np.zeros((len(word2idx)))
pi_robert = np.zeros((len(word2idx)))

#retrieve index of first word of each sentence
for i in range(len(Xtrain_edgar2)):
    pi_edgar[Xtrain_edgar2[i][0]] += 1
pi_edgar += 1 #add-1 smoothing
pi_edgar /= np.sum(pi_edgar) 

#robert
for i in range(len(Xtrain_robert2)):
    pi_robert[Xtrain_robert2[i][0]] += 1
pi_robert += 1
pi_robert /= np.sum(pi_robert) 

#compute state prob. matrix
#rows = prev word, #col = current word
A_edgar = np.zeros((len(word2idx), len(word2idx)))
A_robert = np.zeros((len(word2idx), len(word2idx)))

for s in Xtrain_edgar2:
    for i in range(1,len(s)):
        prevword = s[i-1]
        curword = s[i]
        A_edgar[prevword, curword] += 1
A_edgar += 1
A_edgar /= np.sum(A_edgar, axis = 1) #along axis 1 

#robert
for s in Xtrain_robert2:
    for i in range(1,len(s)):
        prevword = s[i-1]
        curword = s[i]
        A_robert[prevword, curword] += 1
A_robert += 1
A_robert /= np.sum(A_robert, axis = 1) 



#Bayes model
class myBayes():
    def __init__(self, word2idx, pi, a):
        self.word2idx = word2idx
        self.pi = pi 
        self.a = a
    
    def fit(self, Ytrain):
        
        self.K = len(np.unique(Ytrain)) #K classes
        
        #calculate log of priors
        self.p_k = np.zeros((self.K))
        for i in range(self.K):
            self.p_k[i] = len(Ytrain[Ytrain==i]) / len(Ytrain)
        self.p_k = np.log(self.p_k)
        
    def predict(self, X):
        #the X we pass into predict is still just sentences
        #first, we'll convert all input test sentences into idx
        self.xpred = []

        for i in X:
            s = i.lower()
            s = s.translate(str.maketrans('', '', string.punctuation))
            words = s.split()  
            sen = []
            for w in words:
                    
                #if word in the test data doesnt exist, make it the 'unknown' word key we set earlier
                if w not in self.word2idx.keys():
                    w = 'UnW'               
                sen.append(self.word2idx[w]) 
            self.xpred.append(sen)
        
        #next, apply bayes rule
        # recall p(y=k|x) = p(x|y=k) * p(y=k) (ignore marginal since always a constant)
        # log(p(y=k|x)) = log(p(x|y=k)) + log(p(y=k))
        # we already have log of the priors. focus on remaining expression
        # log(p(x|y=k)) -> log [(pi_x=1) *  T∏t=2 A_s(t-1),s(t)]
        # log(p(x|y=k)) -> log(pi_x=1) + T∑t=2 log(A_s(t-1),s(t)) 
                
        pred = np.zeros((len(X), self.K))
        for k in range(self.K): 
            logpi = np.log(self.pi[k])
            logA = np.log(self.a[k])
            
            for i in range(len(self.xpred)):
                
                logpii = logpi[self.xpred[i][0]]
                logAa = 0
                
                for w in range(1, len(self.xpred[i])):
                    prevword = self.xpred[i][w-1]
                    curword = self.xpred[i][w]
                    logAa += logA[prevword,curword]
                
                pred[i,k] = logpii + logAa + self.p_k[k]
        
        return np.argmax(pred, axis = 1)
    
    def score(self, X, Y):
        pred = self.predict(X)
        return np.mean(pred == Y)



model = myBayes(word2idx, 
                [pi_edgar, pi_robert],
                [A_edgar, A_robert]
                )

model.fit(Ytrain)
model.predict(Xtest)

print(model.score(Xtrain, Ytrain))
print(model.score(Xtest, Ytest))

# F1 score
from sklearn.metrics import f1_score

ypredtrain = model.predict(Xtrain)
ypredtest = model.predict(Xtest)

print(f1_score(Ytrain, ypredtrain))
print(f1_score(Ytest, ypredtest))
