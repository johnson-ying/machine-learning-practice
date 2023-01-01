
#gain intuition about PCA

import numpy as np
import matplotlib.pyplot as plt

#1. confirm that Av = λv is valid

#create random matrix
A = np.array([[1,2],[0,2]])

#eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# recall:
# Av = λv
# or in matrix form:
# AV = VΛ 

Av = A.dot(eigenvectors)
lambdav = eigenvectors.dot(np.diag(eigenvalues))

plt.subplot(131)
plt.plot([0, eigenvectors[0,0]], [0, eigenvectors[1,0]], color = 'black', label = 'vector 1')
plt.plot([0, eigenvectors[0,1]], [0, eigenvectors[1,1]], color = 'black', label = 'vector 2')
plt.legend()
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.title('eigenvectors')

plt.subplot(132)
plt.plot([0, Av[0,0]], [0, Av[1,0]], color = 'black', label = 'vector 1')
plt.plot([0, Av[0,1]], [0, Av[1,1]], color = 'black', label = 'vector 2')
plt.legend()
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.title('A.dot(eigenvectors)')

plt.subplot(133)
plt.plot([0, lambdav[0,0]], [0, lambdav[1,0]], color = 'black', label = 'vector 1')
plt.plot([0, lambdav[0,1]], [0, lambdav[1,1]], color = 'black', label = 'vector 2')
plt.legend()
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.title('eigenvectors.dot(eigenvalues)')

###########
#2. observe how PCA transforms data

#assume you have data N x D 
data = np.array([[1,2],[4,2],[2,3]]) #3 samples with dimensionality 2

#get covariance matrix
data_cov = np.cov(data.T)

#eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(data_cov)

#transformed data is z = XQ where Q is eigenvectors
z = data.dot(eigenvectors)

#test assumption is that magnitude of z remains the same as original data
#i.e. sqrt(x[0,0]^2 + x[0,1]^2) should = sqrt(z[0,0]^2 + z[0,1]^2)
squared_orig = np.sqrt(np.sum(data**2,axis = 1))
squared_z = np.sqrt(np.sum(z**2,axis = 1))
print(squared_orig)
print(squared_z)

#test other assumption that z data is simply a rotation of original data
plt.plot([0, data[0,0]], [0, data[0,1]], color = 'black', label = 'vector 1')
plt.plot([0, data[1,0]], [0, data[1,1]], color = 'black', label = 'vector 2')
plt.plot([0, data[2,0]], [0, data[2,1]], color = 'black', label = 'vector 3')
plt.plot([0, z[0,0]], [0, z[0,1]], color = 'red', label = 'transformed vector 1')
plt.plot([0, z[1,0]], [0, z[1,1]], color = 'red', label = 'transformed vector 2')
plt.plot([0, z[2,0]], [0, z[2,1]], color = 'red', label = 'transformed vector 3')
plt.legend()
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.title('original data vs. transformed data')
