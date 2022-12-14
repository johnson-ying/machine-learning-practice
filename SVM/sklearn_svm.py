# classification using sklearn svm API

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.svm import SVC

# load MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# standardize data between 0 and 1
test_X = test_X / 255
train_X = train_X / 255

# resize to N x D 
train_X = np.resize(train_X, (60000,784))
test_X = np.resize(test_X, (10000,784))

# create model
model = SVC()

# train model
model.fit(train_X, train_y)

# train/test scores
print("train score:", model.score(train_X, train_y), "duration:", datetime.now() - t0)
print("test score:", model.score(test_X, test_y), "duration:", datetime.now() - t0)
