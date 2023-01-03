
#bigram model using a deep 1-hidden layer network
#again, code is very similar to bigrams_neural_net and bigrams_neural_net_2
#goal here is to continue building intuition towards word embeddings
#
#Input is N x 1 (or N x V if data is left in one-hot form)
#1st set of weights is V x D (analogous to word embedding)
#2nd set of weights is D x V

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
X = np.argmax(X, axis = 1)
Y = np.array(Y)
Y = np.argmax(Y, axis = 1)

#create test and train data
idx = np.random.choice(len(X), int(0.8*len(X)))
testidx = [i for i in range(len(X)) if i not in idx]

Xtrain = X[idx]
Ytrain = Y[idx]
Xtest = X[testidx]
Ytest = Y[testidx]

#now, create a neural net
#note, resembles an autoencoder
class ANN():
    def __init__(self, M1, M2):
        self.wi = theano.shared(np.random.randn(M1,M2)/np.sqrt(M1))
        self.wo = theano.shared(np.random.randn(M2,M1)/np.sqrt(M2))
        self.params = [self.wi, self.wo]
    
    def forward(self, X):
        middle = T.tanh(self.wi[X])
        out = T.nnet.softmax(middle.dot(self.wo))
        return out
    
    def fit (self, X, Y, lr = 1e-5, mu = 0.9, n_iter = 20):

        thX = T.ivector('inputs')
        thY = T.ivector('targets')

        out = self.forward(thX)
        
        #cost
        cost = -T.mean( T.log(out[T.arange(thY.shape[0]), thY]) )
        
        grads = T.grad(cost, self.params)
        
        updates = [(p, p - lr * g) for p, g in zip(self.params, grads)]

        train = theano.function(inputs = [thX, thY],
                                updates = updates,
                                outputs = cost,
                                allow_input_downcast=True)
        
        self.costs = []
        #train loop
        for i in range(n_iter):
            c = train(X, Y)
            self.costs.append(c)
            print('iter: ', i, 'cost: ', c)
        
        plt.plot(self.costs)
        

model = ANN(total_vocab, 20) #hidden dimension 20 

#training time speeds up by having a lower dimensional embedding
model.fit(Xtrain, Ytrain, lr = 30, n_iter = 200)

