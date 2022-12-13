
#implement a HMM for discrete observations and multiple sequences
#update eqns for Baum Welch algorithm do not involve phi or gamma 
#https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf p.17

import numpy as np
import matplotlib.pyplot as plt

#create a coin toss problem
#observation is H/T, hidden state is the type of coin
#index 0 = head, 1 = tail
pi = np.array([0.5,0.5])
A = np.array([[0.3,0.7], [0.6,0.4]])
B = np.array([[0.8,0.2], [0.4,0.6]])

M, K = B.shape

#create 50 random sequences
X = []
N = 50
length = 30
for i in range(N):
    sequence = []
    initial = np.random.choice(M, size = 1, p = pi)[0]
    sequence.append(np.random.choice(K, size = 1, p = B[initial,:])[0])
    state = initial
    for t in range(1,N):
        state = np.random.choice(M, size = 1, p = A[state,:])[0]
        sequence.append(np.random.choice(K, size = 1, p = B[state,:])[0]) 
    X.append(sequence)

#HMM
class myHMM:
    def __init__(self, M, K):
        #initialize random 
        self.pi = np.zeros((M)) + 1/M 
        self.A = np.random.random((M,M))
        self.A /= np.sum(self.A, axis = 1, keepdims=True)
        self.B = np.random.random((M,K))
        self.B /= np.sum(self.B, axis = 1, keepdims=True)
        
    def fit(self, X, n_iter):
        N = len(X)
        
        likelihoods = []
        
        for _ in range(n_iter):
            self.probs = []
            self.alphas = []
            self.betas = []
            #forward backward algo
            for s in X:
                T = len(s)
                alpha = []
                
                #initialization step
                alpha.append(self.pi * self.B[:,s[0]]) #1 x M 
                #induction steps
                for t in range(1, T):
                    tmp = alpha[-1].dot(self.A) * self.B[:,s[t]] #1 x M
                    alpha.append(tmp)
                #termination step
                self.probs.append(alpha[-1].sum()) #p(x)
                self.alphas.append(alpha)
                
                #backward algo
                beta = []        
                #initialization step
                beta.append(np.ones((M)))   
                #induction steps
                for t in range(T-2, -1, -1):
                    tmp = self.A.dot( self.B[:,s[t+1]] * beta[-1]) #1 x M
                    # print(tmp)
                    beta.append(tmp)
                beta.reverse()
                self.betas.append(beta)
            
            #calculate log likelihood for each iteration
            likelihoods.append(np.sum(np.log(self.probs)))
            
            # update pi, A and B
            self.pi = 0
            for n in range(N):
                self.pi += self.alphas[n][0] * self.betas[n][0] / self.probs[n]
            self.pi /= N
           
            # self.pi = np.sum((self.alphas[n][0] * self.betas[n][0])/self.probs[n] for n in range(N)) / N 
           
            #update A
            a_num = np.zeros((M,M))
            a_den = np.zeros((M,1))
            for n in range(N):
                alpha = self.alphas[n]
                beta = self.betas[n]
                prob = self.probs[n]
                T = len(X[n])
                
                #denominators
                a_den += np.sum( np.array(alpha[:-1]) * np.array(beta[:-1]), axis = 0, keepdims=True).T / prob #1 x M 
                                
                #numerators
                a_num_indiv = np.zeros((M,M)) 
                for i in range(M):
                    for j in range(M):
                        for t in range(T-1):
                            x_t_1 = X[n][t+1]
                            a_num_indiv[i,j] += alpha[t][i] * self.A[i,j] * self.B[j,x_t_1] * beta[t+1][j]
                a_num_indiv /= prob
                a_num += a_num_indiv     
            self.A = a_num / a_den
                
            #update B
            b_num = np.zeros((M,K))
            b_den = np.zeros((M,1))
            for n in range(N):
                alpha = self.alphas[n]
                beta = self.betas[n]
                prob = self.probs[n]
                T = len(X[n])
                
                #denominators
                b_den += np.sum( np.array(alpha) * np.array(beta), axis = 0, keepdims=True).T / prob #1 x M 
                
                #numerators
                b_num_indiv = np.zeros((M,K)) 
                for j in range(M):
                    for k in range(K):
                        for t in range(T):
                            if X[n][t] == k:
                                b_num_indiv[j,k] += alpha[t][j] * beta[t][j]   
                b_num_indiv /= prob
                b_num += b_num_indiv     
            self.B = b_num / b_den
            
        plt.plot(likelihoods)
        
    
    
model = myHMM(M, K)

model.fit(X, 50)  
    
print(model.A)
print(model.B)
print(model.pi)
      
