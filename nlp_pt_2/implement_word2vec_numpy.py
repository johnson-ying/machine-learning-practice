
#implement skip-gram and word2vec in numpy
#note: unoptimized training, just made sure general structure worked

import numpy as np
import matplotlib.pyplot as plt
import string
import random

#load data
from nltk.corpus import brown
lines = brown.sents()
lines = lines[:1000] #work on subset of data

def tokenize(s):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation)) #remove punctuations
    words = s.split()
    return words

#word2idx
word2idx = {}
total_vocab = 0

for s in lines:
    #tokenize 
    s = str(s)
    words = tokenize(s)
    #fill word2idx
    for w in words:
        if w not in word2idx.keys():
            word2idx[w] = total_vocab
            total_vocab += 1

#negative sampling dist
p_w = np.zeros((total_vocab))
for s in lines:
    s = str(s)
    words = tokenize(s)
    for w in words:
        word = word2idx[w]
        p_w[word] += 1

p_w = p_w**0.75
p_w = p_w / np.sum(p_w)

#subsampling dist - probability of dropping a word from a sentence
thres = 8e-5
p_drop_w = 1 - np.sqrt(thres/p_w)
p_drop_w[p_drop_w<0] = 0

#subsample a sentence by dropping words randomly
def subsample(s, p_drop_w):
    s = str(s)
    words = tokenize(s)
    subsampled = []
    for w in words:
        word = word2idx[w]
        
        #get probability of dropping
        prob = np.random.binomial(1, p_drop_w[word])
        
        #if not dropped, add to subsampled
        if prob == 0:
            subsampled.append(word)
    return subsampled

#
#
#

def sigmoid(x):
    return 1/ (1+ np.exp(-x))

#function for SGD, where the label is passed in as 1 or 0 (binary classification)
def SGD(word, context, lr, label):
    #forward propagation 
    #wi[word] - indexing wi by input word - 1 x D
    #wo[:,context] - indexing wo by column by number of context words - D x context_size
    probs = sigmoid(wi[word].dot(wo[:,context])) #1 x context size     
    
    #gradients
    grad_wo = np.outer(wi[word], probs - label) #D x N
    grad_wi = wo[:,context].dot(probs - label) #D x N dot N x 1 = D size
    
    wo[:,context] -= lr * grad_wo
    wi[word] -= lr * grad_wi
    
    #return average cost
    return -np.mean((label * np.log(probs + 1e-9) + (1-label) * np.log(1 - probs + 1e-9)))

#
#
#
#train loop
n_iter = 20
window = 1 #number of words context words to left and right of middle word
lr = 1e-1
costs = []

#initialize model weights
D = 50
wi = np.random.randn(total_vocab, D)/5
wo = np.random.randn(D, total_vocab)/5

for i in range(n_iter):
    
    
    
    for ss in range(len(lines)):
        s = lines[ss]
        
        sentence = subsample(s, p_drop_w)
        if len(sentence) < 3:
            continue
        
        cost = 0.
        #loop through each word in sentence, kind of like stochastic gradient descent
        for n in range(len(sentence)):
            
            if np.isnan(n):
                continue
            
            middle_word = sentence[n]
            left_idx = np.maximum(0, n-2)
            right_idx = np.minimum(len(sentence)-1, n+2)
            
            #grab context words
            context = []
            if left_idx >= 0:
                context += sentence[left_idx:n]
                
            if right_idx <= len(sentence)-1:
                context += sentence[n+1:right_idx+1]
            
            #sample a negative word
            neg_word = np.random.choice(np.array(list(word2idx.values())), p = p_w)

            #stochastic gradient descent - see above function
            c = SGD(middle_word, context, lr, label = 1)
            c = SGD(neg_word, context, lr, label = 0)
               
            cost += c
        
        if ss % 25 == 0:
            costs.append(cost/len(sentence))
            print("iter: ", i, "sentence #: ", ss, "cost: ", cost/len(sentence))
    lr /= 1.1

#plt costs - not optimized
plt.plot(costs)
