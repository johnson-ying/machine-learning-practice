
#implement CBOW (continuous bag of words)
#context size of 2 words on either side

import numpy as np
import matplotlib.pyplot as plt
import string
import theano
import theano.tensor as T
import random

#load data
edgar = open('edgar_allan_poe.txt')
lines = edgar.readlines()
lines2 = [item for item in lines if len(item)>4] #remove lines that are less than length 4

def tokenize(s):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation)) #remove punctuations
    words = s.split()
    return words

#word2idx
word2idx = {}
total_vocab = 0

for s in lines2:
    #tokenize 
    words = tokenize(s)
    
    #fill word2idx
    for w in words:
        if w not in word2idx.keys():
            word2idx[w] = total_vocab
            total_vocab += 1

#create inputs and targets
X = [] #prev word - input
Y = [] #curr word - target

for s in lines2:
    words = tokenize(s)
    
    if len(words)<=5:
        continue
    
    else:
        for n in range(2, len(words)-2):
            prev_word2 = word2idx[words[n-2]]
            prev_word1 = word2idx[words[n-1]]
            next_word1 = word2idx[words[n+1]]
            next_word2 = word2idx[words[n+2]]
            
            curr_word = word2idx[words[n]]
            
            X.append(np.array([prev_word2, prev_word1, next_word1, next_word2]))
            Y.append(curr_word)

#now, create a neural net
class ANN():
    def __init__(self, M1, M2):
        self.wi = theano.shared(np.random.randn(M1,M2)/np.sqrt(M1))
        self.wo = theano.shared(np.random.randn(M2,M1)/np.sqrt(M2))
        self.params = [self.wi, self.wo]
    
    def forward(self, X):
        middle = self.wi[X] #4 x D
        middle = T.mean(middle, axis = 0, keepdims = True) #1 x D - the average of all 4 embeddings
        out = T.nnet.softmax(middle.dot(self.wo)) #1 x V
        return out
    
    def fit (self, X, Y, lr = 1e-5, mu = 0.9, n_iter = 20):

        thX = T.ivector('inputs')
        thY = T.iscalar('targets')

        out = self.forward(thX)
        
        #cost
        #cross entropy on a single sample at a time
        cost = -T.log(out[0, thY])
        
        grads = T.grad(cost, self.params)
        
        updates = [(p, p - lr * g) for p, g in zip(self.params, grads)]

        train = theano.function(inputs = [thX, thY],
                                updates = updates,
                                outputs = cost,
                                allow_input_downcast=True)
        
        self.costs = []
        #train loop
        for i in range(n_iter):
            c_per_i = 0.
            
            #shuffle
            c = list(zip(X, Y))
            random.shuffle(c)
            x_shuf, y_shuf = zip(*c)
            
            for j in range(len(X)): #X in this case is a list - so train sample by sample
                inp = X[j]
                targ = Y[j]
                c = train(inp, targ)
                c_per_i += c
                print('iter: ', i, 'sample: ', j, 'cost: ', c)
                self.costs.append(c)
            # print('iter: ', i, 'cost: ', c_per_i/j)
        
        plt.plot(self.costs)
        

model = ANN(total_vocab, 40) #hidden dimension 40 

model.fit(X, Y, lr = 0.3, n_iter = 5)

#convergence
plt.plot(model.costs)
