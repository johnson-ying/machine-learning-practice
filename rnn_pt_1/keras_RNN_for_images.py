
#use RNN for image classification
#pretend a H x W image is really a T x D sequence

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, GlobalMaxPool1D
from keras.datasets import fashion_mnist

#data already in shape N x T x D
(Xtrain, Ytrain), (Xtest, Ytest) = fashion_mnist.load_data()

K = len(set(Ytrain))
T, D = Xtrain.shape[1], Xtrain.shape[2]


#model - just default to tanh in LSTM for GPU speedup
i = Input(shape = (T,D))
x = LSTM(100, activation = 'tanh', return_sequences=False)(i) #N x T x 100
# x = GlobalMaxPool1D()(x) #N x T
# x = Dense(50, activation = 'relu')(x)
x = Dense(K, activation = 'softmax')(x)

model = Model(inputs = i, outputs = x)

model.compile(
  loss='SparseCategoricalCrossentropy',
  optimizer='Adam',
  metrics=['accuracy']
)

r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=10, batch_size=64)

# loss
plt.plot(r.history['loss'], label='train')
plt.plot(r.history['val_loss'], label='test')
plt.legend()
