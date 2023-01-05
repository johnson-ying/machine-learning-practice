
#implement GloVe in numpy in 2 methods
# 1. alternating least squares
# 2. gradient descent

import numpy as np
import matplotlib.pyplot as plt
import string
import random

#load data
from nltk.corpus import brown
lines = brown.sents()
lines = lines[:5000] #work on subset of data to save time

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
            
#create V x V context distance matrix
X = np.zeros((total_vocab, total_vocab))

for s in lines:
    s = str(s)
    words = tokenize(s)
    
    for n in range(len(words)):
        
        if n == len(words) -1:
            continue
        
        c = 1
        for n2 in range(n+1, len(words)):
            X[word2idx[words[n]], word2idx[words[n2]]] += 1/c
            c += 1
            
#weigh each i and j entry by its x value
#if x value > Xmax, give value of 1, where Xmax is 100
#if x value < Xmax, give value of x**alpha, where alpha is 0.75

#no values greater than 100, so just square to 0.75

Xmax = 100.
alpha = 0.75

fX = np.zeros((total_vocab, total_vocab))
fX[X < Xmax] = (X[X < Xmax] /Xmax) ** alpha
fX[X >= Xmax] = 1

#add 1 to every entry and then take log
X += 1
X = np.log(X) #<- targets 
       
#
#
#matrix factorization
# Xhat = w.dot(u) + b + c + mu
# cost function is MSE weighted by fX plus all the regularization
# fX * (X - Xhat)**2 + reg * (||w||**2 + ||u||**2 + ||b||**2 + ||c||**2)

#hidden dims
D = 30

#total dims
V = total_vocab

#factored weights
w = np.random.randn(V, D)
u = np.random.randn(D, V)

#3 bias terms
b = np.random.randn((V))
c = np.random.randn((V))

#global avg bias term
mu = np.mean(np.mean(X)) 

reg = 1e-4 #regularization constant

costs = []
#train loop
for t in range(6):
    #alternating least squares method 
    #use np.linalg.solve(a, b)
    
    #update w
    for i in range(V):
        A = reg * np.eye(D) + (fX[i,:]*u).dot(u.T) #D x D
        bb = (fX[i,:]*(X[i,:] - b[i] - c - mu)).dot(u.T) #gives size 1 x D
        w[i] = np.linalg.solve(A, bb) #gives size 1 x D
    
    #update b
    for i in range(V):
        b[i] = 1/(np.sum(fX[i,:] + reg)) * fX[i,:].dot(X[i,:] - w[i].dot(u) - c - mu)
      
    #update u
    for j in range(V):
        A = reg * np.eye(D) + (fX[:,j]*w.T).dot(w) #D x D
        bb = (fX[:,j]*(X[:,j] - b - c[j] - mu)).dot(w) #gives size 1 x D
        u[:,j] = np.linalg.solve(A, bb) #gives size 1 x D
    
    #update c
    for j in range(V):
        c[j] = 1/(np.sum(fX[:,j] + reg)) * fX[:,j].dot(X[:,j] - w.dot(u[:,j]) - b - mu)            
    
    recreated = w.dot(u) + b.reshape(V,1) + c.reshape(1,V) + mu
    
    cost = np.sum( fX * (X - recreated)**2)
    costs.append(cost)
    print(cost)

#convergence
plt.plot(costs)

#
#
#alternatively, instead of alternating least squares, can also use gradient descent
w = np.random.randn(V, D)
u = np.random.randn(D, V)
b = np.random.randn((V))
c = np.random.randn((V))

costs = []
lr = 3e-3
reg = 1e-2
#train loop
for t in range(10):

    #after 6 epochs, lower lr 
    if t == 5:
        lr = 3e-4    

    #update w
    for i in range(V):
        g_wi = -(fX[i,:] * 2 * (X[i,:] - w[i].dot(u) - b[i] - c - mu)).dot(u.T) #1xV dot VxD
        g_wi += 2 * reg * w[i] #reg 1xD
        w[i] -= lr * g_wi #1 x D
    
    #update b
    for i in range(V):
        g_bi = -2 * fX[i,:].dot(X[i,:] - w[i].dot(u) - b[i] - c - mu) #1x1
        g_bi += 2* reg * b[i]
        b[i] -= lr * g_bi
        
    #update u
    for j in range(V):
        g_uj = -(fX[:,j] * 2 * (X[:,j] - w.dot(u[:,j]) - b - c[j] - mu)).dot(w) #1xV dot VxD
        g_uj += 2 * reg * u[:,j] #reg 1xD
        u[:,j] -= lr * g_uj #1 x D

    #update c
    for j in range(V):
        g_cj = -2 * fX[:,j].dot(X[:,j] - w.dot(u[:,j]) - b - c[j] - mu) #1x1
        g_cj += 2* reg * c[j]
        c[j] -= lr * g_cj
                   
    recreated = w.dot(u) + b.reshape(V,1) + c.reshape(1,V) + mu
    
    cost = np.sum( fX * (X - recreated)**2)
    costs.append(cost)
    print(cost)

#not quite, but could reach with more training epochs
plt.plot(costs)

