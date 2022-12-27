
#practice markov models and genetic algorithm in context of cipher decryption
#data from: https://lazyprogrammer.me/course_files/moby_dick.txt
#1. create substition cipher (ground truth)
#2. create language model on big corpus. unigram + bigram on individual letters, NOT words
#3. create encoding and decoding functions 
#4. create genetic algorithm to search for best decryption model 

import numpy as np
import matplotlib.pyplot as plt
import string
import re
import random 

#import data, split into paragraphs
data = open('moby_dick.txt', encoding="utf8").read()
data = data.split("\n\n")

#create cipher substitution 
alphabet = list(string.ascii_lowercase)
shuffled = list(string.ascii_lowercase)
random.shuffle(shuffled)

cipher = {value: shuf for value, shuf in zip(alphabet, shuffled)}

#initial letter probability, and first order markov letter transition probability
pi = {value: 0 for value in list(string.ascii_lowercase)}
A = {value: {} for value in list(string.ascii_lowercase)}
for i in list(string.ascii_lowercase):
    A[i] = {value: 0 for value in list(string.ascii_lowercase)}

#find index where first letter starts in each block of text
firstletter = {}
for si in range(len(data)):
    s = data[si]
    s = s.lower()
    for i in range(len(s)):
        if s[i] in cipher.keys():
            firstletter[si] = i
            break

for si in range(len(data)):
    if si in firstletter.keys():
        s = data[si]
        s = s.lower()
        for i in range(firstletter[si], len(s)):     
            #first word
            if i == firstletter[si]:
                pi[s[i]] += 1    
            #remaining sentence
            else:
                t_1 = s[i-1]
                t = s[i]
                if t_1 in cipher.keys(): #only work with letters
                    if t in cipher.keys(): 
                        A[t_1][t] += 1
                else:
                    continue
                
#normalize and plus-one smoothing
pi = {value:key+1 for value, key in pi.items()}
totalcount = sum(pi.values())
pi = {value:key/totalcount for value, key in pi.items()}

for i in A.keys():
    A[i] = {value:key+1 for value, key in A[i].items()}
    totalcount = sum(A[i].values())
    A[i] = {value:key/totalcount for value, key in A[i].items()}

#encoding and decoding functions
def encoding(s):
    encoded = ''
    s = s.lower()
    for i in range(len(s)):
        if s[i] in cipher.keys():
            encoded += cipher[s[i]]
        else:
            encoded += s[i]
    return encoded
    
def decoding(s, dec): 
    decoded = ''
    for i in range(len(s)):
        if s[i] in dec.keys():
            decoded += dec[s[i]]
        else:
            decoded += s[i]
    return decoded

#
# genetic algorithm

#calculate probability of a sequence of letters happening using pi and A
def log_likelihood(s, pi, A):
    log = 0
    firstcount = 1
    for i in range(len(s)):
        if s[i] in cipher.keys():  
            #first letter
            if firstcount == 1:
                log += np.log(pi[s[i]])
                firstcount += 1       
            #remaining sentence
            else:
                t_1 = s[i-1]
                t = s[i]
                if t_1 in A.keys():
                    if t in A[t_1].keys():
                        log += np.log(A[t_1][t])
    return log

#take in a list of sentences, and calculate mean log likelihood value
def fitness(s, dec, pi, A):
    encoded = encoding(s)
    decoded = decoding(encoded, dec)
    return log_likelihood(decoded, pi, A)
    



sentencetotrainon = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
'''

#create 20 different random decoding dicts
DNA_pool = []
for i in range(20):
    a = list(string.ascii_lowercase)
    s = list(string.ascii_lowercase)
    random.shuffle(s)
    DNA_pool.append({value: shuf for value, shuf in zip(a, s)})

#parameters and stuff
epochs = 1000
score = []
pii = list(np.repeat(pi,20))
AA = list(np.repeat(A,20))
sentence = [sentencetotrainon]*20

#training
for i in range(epochs):
    print('iteration #', i)
    #create 3 offsprings per parent
    if i > 0:
        offsprings= []
        for d in DNA_pool:
            for _ in range(3):
                #for each offstring, shuffle 2 random dict rows
                values = list(d.values())
                idx = np.random.choice(25, size=2, replace = False)
                d1 = list(d.values())[idx[0]]
                d2 = list(d.values())[idx[1]]
                values[idx[0]] = d2
                values[idx[1]] = d1
                offsprings.append({value: shuf for value, shuf in zip(d.keys(), values)})
        DNA_pool = DNA_pool + offsprings
    
    scores = [fitness(s, d, pp, aa) for s, d, pp, aa in zip(sentence, DNA_pool, pii, AA)]
    score.append(np.mean(scores)) #save scores for later
    
    #sort DNA by score and keep top 5
    # DNA_pool = [x for y, x in sorted(zip(scores, DNA_pool))]
    
    sortit = np.argsort(scores)
    DNA_pool =list(np.array(DNA_pool)[sortit])
    DNA_pool = DNA_pool[-5:]
    
    
#results vary. if log converges to ~ -800, then decoding is pretty good
plt.plot(score) 
    
print(sentencetotrainon)
encoded = encoding(sentencetotrainon)
print('')
print(decoding(encoded, DNA_pool[4]))
