
#review: implement bigram (first order markov) model on pre processed wikipedia data
#txt files not provided, but any corpus txt files or preloaded corpus would do...

import numpy as np
import matplotlib.pyplot as plt
import glob, os
import string

#pull all wiki data
for file in glob.glob("*.txt"):
    with open(file, encoding="utf8") as f:
        lines = f.readlines()

#only look at text longer than certain length
lines = [l for l in lines if len(l) >= 75]

lines2 = []
#pull out indiv sentences
for l in lines:
    lines2 += l.split(".")
lines2 = [l for l in lines2 if len(l) >= 50]

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

#create initial state matrix and bigram transition probability matrix
#use dicts because too many words, would use too much ram if we did numpy array
pi = {}
A = {} #going to be a nested dict

#fill matrices
for s in lines2:
    words = tokenize(s)
    
    for n in range(len(words)):
        #initial word
        if n == 0:
            idx = word2idx[words[n]]
            #if word idx not in dict, initiate it
            if idx not in pi.keys():
                pi[idx] = 0
            pi[idx] += 1
        #following words
        else:
            prev_word = word2idx[words[n-1]]
            curr_word = word2idx[words[n]]
            #if word idx not in dict, initiate a nested dict
            if prev_word not in A.keys():
                A[prev_word] = {}
            if curr_word not in A[prev_word].keys():
                A[prev_word][curr_word] = 0
            A[prev_word][curr_word] += 1
    
#convert all values to probabilities
totalcount = sum(pi.values())
pi = {key: value/totalcount for key, value in pi.items()}

for key in A.keys():
    totalcount = sum(A[key].values())
    A[key] = {key: value/totalcount for key, value in A[key].items()}
    
