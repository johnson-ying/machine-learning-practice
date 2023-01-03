
#improve bigrams_neural_net by directly indexing weight matrix
#very similar to last code, just a couple changes
#i.e. treating the weight matrix as an embedding

import numpy as np
import matplotlib.pyplot as plt
import string
import theano
import theano.tensor as T

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

#create bigram transition probability matrix
A = np.zeros((total_vocab, total_vocab))

#fill matrices
for s in lines2:
    words = tokenize(s)
    for n in range(1, len(words)):
        prev_word = word2idx[words[n-1]]
        curr_word = word2idx[words[n]]
        A[prev_word, curr_word] += 1

#add tiny bit of smoothing
A += 1e-5

#convert all values to probabilities and then log it
totalcount = np.sum(A, axis = 1, keepdims = True)
A = A/totalcount
logA = np.log(A) 


############################################################

#now, implement neural network
X = [] #prev word - input
Y = [] #curr word - target

for s in lines2:
    words = tokenize(s)
    for n in range(1, len(words)):
        #put words into 1 x total_vocab size array
        prev_word = np.zeros((total_vocab)) 
        prev_idx = word2idx[words[n-1]]
        prev_word[prev_idx] += 1
        
        curr_word = np.zeros((total_vocab)) 
        curr_idx = word2idx[words[n]]
        curr_word[curr_idx] += 1
        
        X.append(prev_word)
        Y.append(curr_word)
        
#convert into N 
X = np.array(X)
X = np.argmax(X, axis = 1) #<-- now, input will no longer be N x V.. but just N x 1
Y = np.array(Y)
Y = np.argmax(Y, axis = 1)

#create test and train data
idx = np.random.choice(len(X), int(0.8*len(X)))
testidx = [i for i in range(len(X)) if i not in idx]

Xtrain = X[idx]
Ytrain = Y[idx]
Xtest = X[testidx]
Ytest = Y[testidx]

#neural net

class ANN():
    def __init__(self, M1, M2, logA):
        self.w = theano.shared(np.random.randn(M1,M2)/np.sqrt(M1))
        self.params = [self.w]
        self.logA = theano.shared(logA)
    
    def forward(self, X):
        return T.nnet.softmax(self.w[X]) #no longer dot product, but directly indexing
    
    def forward_logA(self, X): 
        return T.nnet.softmax(self.logA[X]) #no longer dot product, but directly indexing
    
    def fit (self, X, Y, lr = 1e-5, mu = 0.9, n_iter = 20):

        thX = T.ivector('inputs') #<-- no longer matrix, but just a vector
        thY = T.ivector('targets')
        
        out = self.forward(thX)
    
        out_logA = self.forward_logA(thX)
        
        #cost
        cost = -T.mean( T.log(out[T.arange(thY.shape[0]), thY]) )
        logAcost = -T.mean( T.log(out_logA[T.arange(thY.shape[0]), thY]) )
        
        grads = T.grad(cost, self.params)
        
        updates = [(p, p - lr * g) for p, g in zip(self.params, grads)]

        train = theano.function(inputs = [thX, thY],
                                updates = updates,
                                outputs = cost,
                                allow_input_downcast=True)
        
        get_logA_cost = theano.function(inputs = [thX, thY],
                                outputs = logAcost,
                                allow_input_downcast=True)
        
        self.costs = []
        self.logAcosts = []
        #train loop
        for i in range(n_iter):
            
            c = train(X, Y)
            self.costs.append(c)
            
            logAc = get_logA_cost(X, Y)
            self.logAcosts.append(logAc)
            
            print('iter: ', i, 'cost: ', c, 'target cost: ', logAc)
        
        plt.plot(self.costs)
        plt.plot(self.logAcosts)
        

model = ANN(total_vocab, total_vocab, logA)

#my theano isnt optimized for speed so didnt test, but should theoretically be faster 
model.fit(Xtrain, Ytrain, lr = 30, n_iter = 500)


#visualize learned bigram prob matrix vs. ground truth
plt.subplot(121)
plt.imshow(model.w.eval())
plt.jet()
plt.clim([0,0.5])

plt.subplot(122)
plt.imshow(A)
plt.jet()
plt.clim([0,0.5])

