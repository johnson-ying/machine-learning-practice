
#PCA on mnist data

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

Xtrain = Xtrain.reshape((len(Xtrain), -1)) #reshape to N x D 

#normalize data
Xtrain, Xtest = Xtrain/255, Xtest/255

#covariance matrix 
cov = np.cov(Xtrain.T)

#eigenvalues, eigenvectors
lambdas, Q = np.linalg.eig(cov)

#sort lambdas
idx = np.argsort(-lambdas)
lambdas = lambdas[idx]
lambdas = np.maximum(0, lambdas) #convert small negative values to 0
Q = Q[:,idx] #sort eigenvectors

#transformed data
z = Xtrain.dot(Q)

#plot first 2 dimensions of transformed data
plt.scatter(z[:,0], z[:,1], s = 30, c = Ytrain)
plt.xlabel('PC1')
plt.ylabel('PC2')

#plot variances
plt.plot(lambdas)

#how many components capture 95% of total variance in data?
total_var = np.sum(lambdas)
cum_var = np.cumsum(lambdas)
np.where(cum_var >= 0.95 * total_var)[0][0]

#90%?
np.where(cum_var >= 0.9 * total_var)[0][0]




########
# sklearn PCA

from sklearn.decomposition import PCA

pca = PCA()
z = pca.fit_transform(Xtrain)

#plot first 2 dimensions of transformed data
plt.scatter(z[:,0], z[:,1], s = 30, c = Ytrain)
plt.xlabel('PC1')
plt.ylabel('PC2')

lambdas = pca.explained_variance_

#plot variances
plt.plot(lambdas)

#how many components capture 95% of total variance in data?
total_var = np.sum(lambdas)
cum_var = np.cumsum(lambdas)
np.where(cum_var >= 0.95 * total_var)[0][0]

#90%?
np.where(cum_var >= 0.9 * total_var)[0][0]
