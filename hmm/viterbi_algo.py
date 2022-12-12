
#implement the viterbi algorithm given pre-computed state transition matrices

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

M = 3 #number of hidden states
K = 3 #number of observations (discrete)
T = 8 #time steps

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

#viterbi algo: find the sequence of hidden states that best explains observations
def viterbi(s, pi, A, B):
    deltas = np.zeros((T,M)) #store individual probs at each t
    phis = np.zeros((T,M)) #store most likely hidden state at each t
    x_t = s[0] #initial observation
    
    #initialization step
    deltas[0] = np.max(pi * B[:,x_t])
    
    #induction steps
    for t in range(1, len(s)):
        for j in range(M):    
            deltas[t,j] = np.max(deltas[t-1] * A[:,j]) * B[j,s[t]]
            phis[t,j] = np.argmax(deltas[t-1] * A[:,j])
    
    #termination step
    states = np.zeros((T)).astype(int)
    states[-1] = np.argmax(deltas[-1]) #final state
    for t in range(T-2, -1, -1):
        states[t] = phis[t+1, states[t+1]]
    return states
    
hidden = viterbi(sequence, pi, A, B)
