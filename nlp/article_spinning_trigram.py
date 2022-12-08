
#article spinning using trigram 
#a word is conditioned on past and future word: p(w_t | w_t-1, w_t+1)
#store probabilities inside of dictionaries
#http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import string

positive = BeautifulSoup(open('positive.review').read())
positive = positive.findAll('review_text')
positive = [s.text for s in positive]

def tokenize(s):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation)) #remove punctuations
    words = s.split()
    return words

def add2trigram(d, beforeafter, middle):
    if beforeafter not in d.keys():
        d[beforeafter] = {} #a nested dict
    if middle not in d[beforeafter].keys():
        d[beforeafter][middle] = 0
    d[beforeafter][middle] += 1
        
#create trigram dict probabilities
trigram = {}
for s in positive:
    words = tokenize(s)
    for i in range(1,len(words)-1):
        before = words[i-1]
        middle = words[i]
        after = words[i+1]
        
        add2trigram(trigram, (before,after), middle)

#normalize all probabilities in each conditional probability
for key in trigram.keys():
    totalcount = sum(trigram[key].values())
    trigram[key] = {key: value/totalcount for key, value in trigram[key].items()}
    
def sample_a_word(d,key):
    word = np.random.choice(len(d[key]), size = 1, p = np.fromiter(d[key].values(), dtype=float))[0]
    word = list(d[key])[word]
    return word

def spin_article(article):
    
    orig_w_punc = article.split() #save original with punctuation
    
    words = tokenize(article)
    length = len(words)
    
    for i in range(1, len(words)-1):
        before = words[i-1]
        after = words[i+1]
        
        #20% chance that we'll sample a new word
        if(np.random.choice(100,size=1)[0] <= 20):
            if (before, after) in trigram.keys():
                newword = sample_a_word(trigram, (before, after))
                orig_w_punc[i] = newword
            
            #specific scenario if we were sampling a new word when the previous word was JUST sampled
            #the new key wouldnt exist, so skip
            else:
                continue
            
    return ' '.join(orig_w_punc)

#test it out
idx = np.random.choice(len(positive),size=1)[0]
print(positive[idx])
print('')
print(spin_article(positive[idx]))

