
#gain intuition about SVD

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

np.random.seed(45)

#Try out SVD eqn 
# X = U * S * V.T

#create random data X and mean center
X = np.random.randn(5,4)
mu = np.mean(X, axis = 0, keepdims=True)
X -= mu

#Calculate U which are eigenvectors of X^T * X
Vlambda, V = np.linalg.eig(X.T.dot(X))
idx = np.argsort(-Vlambda)
Vlambda = Vlambda[idx]
Vlambda = np.maximum(0, Vlambda) #set very small neg values to 0
V = V[:,idx]

#Calculate V which are eigenvectors of X * X^T
Ulambda, U = np.linalg.eig(X.dot(X.T))
idx = np.argsort(-Ulambda)
Ulambda = Ulambda[idx]
Ulambda = np.maximum(0, Ulambda) #set very small neg values to 0
# Ulambda = Ulambda[:-1] #drop last one
U = U[:,idx]
# U = U[:,:-1] #drop last one b/c shape must be N x D

#Confirm that the eigenvalues are the same for either U or V
print(np.sum(Ulambda) - np.sum(Vlambda)) #basically 0

#S is the square root of eigenvalues 
S = np.sqrt(np.diag(Ulambda))
S = S[:,:-1] #drop last column

#same, aside from flipped signs
print('Re-created')
print(U.dot(S).dot(V.T))
print('Original')
print(X)

#########
# using sklearn PCA vs. SVD
from sklearn.decomposition import PCA, TruncatedSVD

(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
Xtrain = Xtrain.reshape((len(Xtrain), -1)) #reshape to N x D 

#normalize data
Xtrain, Xtest = Xtrain/255, Xtest/255

#mean center data
mu = np.mean(Xtrain, axis = 0, keepdims=True)
Xtrain -= mu


pca = PCA(2)
z1 = pca.fit_transform(Xtrain)

svd = TruncatedSVD(2)
z2 = svd.fit_transform(Xtrain)

plt.subplot(121)
plt.scatter(z1[:,0], z1[:,1], s = 30, c = Ytrain)
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(122)
plt.scatter(z2[:,0], z2[:,1], s = 30, c = Ytrain)
plt.xlabel('PC1')
plt.ylabel('PC2')
