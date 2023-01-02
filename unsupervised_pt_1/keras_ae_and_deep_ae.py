
#autoencoders in keras

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import fashion_mnist
from keras.optimizers import Adam

(Xtrain, Ytrain), (Xtest, Ytest) = fashion_mnist.load_data()

Xtrain = np.reshape(Xtrain, (len(Xtrain), -1))
Xtest = np.reshape(Xtest, (len(Xtest), -1))

Xtrain = Xtrain/255
Xtest = Xtest/255

N, D = Xtrain.shape


# Scenario #1: Regular 1-layer autoencoder

hidden_dim = 100 #middle layer is 100 dim

#model
i = Input(shape = (D,))
encoder = Dense(hidden_dim, activation = 'sigmoid')(i)
decoder = Dense(D, activation = 'sigmoid')(encoder)

model = Model(inputs = i, outputs = decoder)

model.compile(
  loss='mse',
  optimizer='Adam',
)

r = model.fit(Xtrain, Xtrain, epochs=20, batch_size=64)

# loss
plt.plot(r.history['loss'], label='train')
plt.legend()

#observe recreation
idx = np.random.randint(0, len(Xtrain)-2)
pred = model.predict(Xtrain[idx:idx+1])
pred = pred.reshape(28,28)

plt.subplot(121)
plt.imshow(Xtrain[idx:idx+1].reshape(28,28))
plt.gray()

plt.subplot(122)
plt.imshow(pred)

###############################################################################

# Scenario #2: Deep autoencoder with middle dim = 2

#model
i = Input(shape = (D,))
x = Dense(300, activation = 'relu')(i)
x = Dense(100, activation = 'relu')(x)
encoded = Dense(2, activation = 'relu')(x)

x = Dense(100, activation = 'relu')(encoded)
x = Dense(300, activation = 'relu')(x)
x = Dense(D, activation = 'sigmoid')(x)

model = Model(inputs = i, outputs = x)

model.compile(
    loss='mse',
   # loss='binary_crossentropy',
  optimizer='Adam',
)

r = model.fit(Xtrain, Xtrain, epochs=20, batch_size=200)

# loss
plt.plot(r.history['loss'], label='train')
plt.legend()

#get latent representation
model2 = Model(inputs = i, outputs = encoded)
latent = model2.predict(Xtrain)
plt.scatter(latent[:,0], latent[:,1], c = Ytrain)
plt.jet()

#observe recreation
idx = np.random.randint(0, len(Xtrain)-2)
pred = model.predict(Xtrain[idx:idx+1])
pred = pred.reshape(28,28)

plt.subplot(121)
plt.imshow(Xtrain[idx:idx+1].reshape(28,28))
plt.gray()

plt.subplot(122)
plt.imshow(pred)

