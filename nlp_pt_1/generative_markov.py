# build a 2nd-order markov text generation model on robert frost poetry
#data from:
#https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt
#
# instead of np arrays, store transition probabilities inside of dictionaries

import numpy as np
import matplotlib.pyplot as plt
import string

#import data
robert = open('robert_frost.txt')
robert = robert.readlines()
robert = [item for item in robert if len(item)>4] #remove lines that are less than length 4
data = []
for s in robert:
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation)) #remove punctuations
    words = s.split()
    if len(words) >= 3: #only append sentence if it contains more than 3 words
        data.append(s)

#probability dicts
pi = {}
A1 = {}
A2 = {}

#first word dictionary
totalcount = 0
for s in data:
    words = s.split()
    w = words[0]
    pi[w] = pi.get(w, 0) + 1
    totalcount += 1
pi = {key: value/totalcount for key, value in pi.items()}

#second word dictionary
#a nested dictionary - master dict contains first word, and each word contains a dict for second word transition probs.
for s in data:
    words = s.split()
    first = words[0]
    second = words[1]
    if first not in A1.keys():
            A1[first] = {} #create a nested dict
    if second not in A1[first].keys():
            A1[first][second] = 0    
    A1[first][second] += 1   
#convert to probabilities
for n in A1.keys():
    totalcount = np.sum([value for key, value in A1[n].items()])
    A1[n] = {key: value/totalcount for key, value in A1[n].items()}

#third word and beyond dictionary
#nested dict inside a nested dic inside a master dict
for s in data:
    words = s.split()
    t_2 = words[0]
    t_1 = words[1]
    for i in range(2, len(words)):   
        t_2 = words[i-2]
        t_1 = words[i-1]
        t = words[i]
        
        if t_2 not in A2.keys():
            A2[t_2] = {} #nested dict
        if t_1 not in A2[t_2].keys():
            A2[t_2][t_1] = {} #nested dict
        if t not in A2[t_2][t_1].keys():
            A2[t_2][t_1][t] = 0 #count
            
        A2[t_2][t_1][t] += 1 #add count
        
        #at the last word, need to add an "end of sentence" term to mark end of sentence
        if i == len(words)-1:
            if t_1 not in A2.keys():
                A2[t_1] = {} #nested dict
            if t not in A2[t_1].keys():
                A2[t_1][t] = {} #nested dict    
            if 'EOS' not in A2[t_1][t].keys():
                A2[t_1][t]['EOS'] = 0 #count
                
            A2[t_1][t]['EOS'] += 1 #end of sentence
        
#convert to probabilities
for n in A2.keys():
    for nn in A2[n].keys():
        totalcount = np.sum([value for key, value in A2[n][nn].items()])
        A2[n][nn] = {key: value/totalcount for key, value in A2[n][nn].items()}

#generate poetry 
def generate_poetry(numlines):
    
    poetry = []
    
    for ii in range(numlines): 
        sen = ''
        first = np.random.choice(len(pi), size = 1, p = np.fromiter(pi.values(), dtype=float))[0]
        first = list(pi)[first]
        second = np.random.choice(len(A1[first]), size = 1, p = np.fromiter(A1[first].values(), dtype=float))[0]
        second = list(A1[first])[second]
            
        sen = first + ' ' + second
        
        #keep iterating remaining line until encounter an EOS
        while True:
            third = np.random.choice(len(A2[first][second]), size = 1, p = np.fromiter(A2[first][second].values(), dtype=float))[0]
            third = list(A2[first][second])[third]
            
            if(third == 'EOS'):
                break
            
            sen += ' ' + third
            
            first = second
            second = third  
            
        poetry.append(sen)  
        
    return poetry
    
po = generate_poetry(numlines = 6)
print(po)
