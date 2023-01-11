
#bidirectional RNN for images

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional, GlobalMaxPool1D, Permute, Concatenate
from keras.datasets import fashion_mnist

(Xtrain, Ytrain), (Xtest, Ytest) = fashion_mnist.load_data()

Xtrain = Xtrain / 255
Xtest = Xtest / 255

N, T, D = Xtrain.shape
K = len(set(Ytrain))

#model
i = Input(shape = (T,D))
lstm1 = Bidirectional(LSTM(64, return_sequences = True))(i) #N x T x 2*D

#create a second branch that rotates the image and then does another BiRNN
rotated = Permute((2,1))(i)
lstm2 = Bidirectional(LSTM(64, return_sequences = True))(rotated) #N x T x 2*D

#concat all data 
concat = Concatenate(axis=-1)([lstm1, lstm2]) #N x T x 4*D

x = GlobalMaxPool1D()(concat) #N x 4*D

x = Dense(K, activation = 'softmax')(x)

model = Model(i, x)

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='Adam',
  metrics=['accuracy'],
)

r = model.fit(Xtrain, Ytrain, validation_data = (Xtest, Ytest), batch_size=64, epochs=5)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')



# #make sure rotated image worked
# view_concat = Model(i, rotated)

# img = Xtrain[0:1,:,:]
# rot = view_concat(img)

# plt.subplot(121)
# plt.imshow(img[0])
# plt.subplot(122)
# plt.imshow(rot[0])
