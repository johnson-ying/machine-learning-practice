
#practice using beautifulsoup, nltk, and lemmatize
#and writing tokenizer and word2vec functions

import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
import string

lemma = WordNetLemmatizer()
stopwords = [word.rstrip() for word in open('stopwords.txt')]

positive = BeautifulSoup(open('positive.review').read())
positive = positive.findAll('review_text')
positive = [s.text for s in positive]

negative = BeautifulSoup(open('negative.review').read())
negative = negative.findAll('review_text')
negative = [s.text for s in negative]

#tokenizer function
def tokenize(s):
    s = s.lower()
    # s = s.translate(str.maketrans('', '', string.punctuation)) #remove punctuations
    # words = s.split()
    words = nltk.tokenize.word_tokenize(s)
    words = [w for w in words if len(w) > 2]
    words = [lemma.lemmatize(w) for w in words]
    words = [w for w in words if w not in stopwords]
    return words

#word2idx
word2idx = {}
total_vocab = 0

for s in positive:
    #tokenize 
    words = tokenize(s)
    
    #fill word2idx
    for w in words:
        if w not in word2idx.keys():
            word2idx[w] = total_vocab
            total_vocab += 1

for s in negative:
    #tokenize 
    words = tokenize(s)
    
    #fill word2idx
    for w in words:
        if w not in word2idx.keys():
            word2idx[w] = total_vocab
            total_vocab += 1


#convert tokens to words counts within each sentence
def word2vec(sentences, total_vocab):
    allvec = np.zeros((len(sentences), total_vocab))
    for i, s in zip(range(len(sentences)), sentences):
        #tokenize
        words = tokenize(s)
        for w in words:
            j = word2idx.get(w)
            allvec[i, j] += 1 
        #normalize each sentence by total count
        allvec[i] /= len(words)
    return allvec
        
positive_idx = word2vec(positive, total_vocab)
negative_idx = word2vec(negative, total_vocab)



#create train,test sets and run classification

X = np.concatenate((positive_idx, negative_idx), axis = 0)
Y = np.concatenate((np.zeros((len(positive_idx)))+1, np.zeros((len(negative_idx)))), axis = 0)

p = np.random.permutation(len(X))
X = X[p]
Y = Y[p]

idx = np.random.choice(len(X), int(0.8 * len(X)), replace = False)
testidx = [i for i in range(len(X)) if i not in idx]

Xtrain, Ytrain = X[idx], Y[idx]
Xtest, Ytest = X[testidx], Y[testidx]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
model.score(Xtest,Ytest)

from sklearn.model_selection import cross_val_score
model = LogisticRegression()
print(np.mean(cross_val_score(model, X, Y, cv=5)))
