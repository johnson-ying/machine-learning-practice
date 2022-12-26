
#ANN in keras using sequential or functional API 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.datasets import mnist

# load MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# standardize data between 0 and 1
test_X = test_X / 255
train_X = train_X / 255

# resize to N x D 
train_X = np.resize(train_X, (60000,784))
test_X = np.resize(test_X, (10000,784))


N, D = train_X.shape
K = len(set(train_y))



model = Sequential()

model.add(Dense(units = 300, activation = 'relu', input_dim = D))
model.add(Dense(units = 100, activation = 'relu'))
model.add(Dense(units = K, activation = 'softmax'))

model.compile(
  loss='SparseCategoricalCrossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

r = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=10, batch_size=64)

# accuracies
plt.plot(r.history['accuracy'], label='train')
plt.plot(r.history['val_accuracy'], label='test')
plt.legend()

####
# using keras functional API


from keras.models import Model
from keras.layers import Input

i = Input(shape = (D,))
x = Dense(300, activation = 'relu')(i)
x = Dense(100, activation = 'relu')(x)
x = Dense (K, activation = 'softmax')(x)

model = Model(inputs = i, outputs = x)

model.compile(
  loss='SparseCategoricalCrossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

r = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=10, batch_size=64)

# accuracies
plt.plot(r.history['accuracy'], label='train')
plt.plot(r.history['val_accuracy'], label='test')
plt.legend()


