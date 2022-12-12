
#implement the forward-backward algorithm given pre-computed state transition matrices

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

M = 3 #number of hidden states
K = 3 #number of observations (discrete)
T = 3 #time steps

#initial hidden state probability matrix
pi = np.zeros((M)) + 1/M 

#hidden state transition matrix
A = np.random.randint(low=1,high=100,size=(M,M)).astype('float')
A /= np.sum(A, axis = 1, keepdims=True)

#hidden-observation matrix
B = np.random.randint(low=1,high=100,size=(M,K)).astype('float') 
B /= np.sum(B, axis = 1, keepdims=True)

#create random sequence of observations
sequence = np.random.choice(K, size = T, replace = True)

#forward algo
def forward(s, pi, A, B):
    alphas = []
    x_t = s[0] 
    
    #initialization step
    alphas.append(pi * B[:,x_t]) #1 x M 
    
    #induction steps
    for t in range(1, len(s)):
        tmp = alphas[t-1].dot(A) * B[:,s[t]] #1 x M
        alphas.append(tmp)
    
    #termination step
    return alphas, alphas[-1].sum()

alphas, p = forward(sequence, pi, A, B)

#backward algo
def backward(s, pi, A, B):
    betas = []
    x_t = s[-1] 
    T = len(s) - 1
    
    #initialization step
    betas.append(1)
    
    #induction steps
    for t in range(T-2, -1, -1)
        tmp = A.dot( B[:,s[t]]) * betas[-1] #1 x M
        betas.append(tmp)
    betas.reverse()
    return betas

betas = backward(sequence, pi, A, B)
