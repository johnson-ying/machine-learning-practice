
#visualize the learned features in CNN architectures

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D
from keras.datasets import fashion_mnist
from keras.optimizers import Adam

(Xtrain, Ytrain), (Xtest, Ytest) = fashion_mnist.load_data()

Xtrain = np.expand_dims(Xtrain, -1)
Xtest = np.expand_dims(Xtest, -1)


K = len(set(Ytrain))

#~93% accuracy
i = Input(shape = Xtrain[0].shape)
conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(i)
x = BatchNormalization()(conv1)
conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(conv2)
x = MaxPooling2D((2,2))(x)
conv3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(conv3)
conv4 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(conv4)
x = MaxPooling2D((2,2))(x)
conv5 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(conv5)
conv6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(conv6)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(K, activation = 'softmax')(x)


model = Model(inputs = i, outputs = x)

model.compile(
  loss='SparseCategoricalCrossentropy',
  # optimizer = optimizer,
  optimizer='Adam',
  metrics=['accuracy']
)

r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=10, batch_size=64)

# accuracies
plt.plot(r.history['accuracy'], label='train')
plt.plot(r.history['val_accuracy'], label='test')
plt.legend()







#visualize learned features
#create separate model to retrieve indiv layers
model_learned_features = Model(inputs = i, outputs = [conv1, conv2, conv3, conv4, conv5, conv6])

#get random image
randomid = np.random.randint(0, len(Xtest))
img = np.reshape(Xtest[randomid], (1,28,28,1))
plt.imshow(img[0,:,:,0])
plt.gray()

#get features 
f1,f2,f3,f4,f5,f6 = model_learned_features.predict(img)

#first conv
fig1 = plt.figure("Conv 1")
for i in range(f1.shape[-1]):
    plt.subplot(8,8,i+1)
    plt.imshow(f1[0,:,:,i])
    plt.axis('off') 

#second conv
fig2 = plt.figure("Conv 2")
for i in range(f2.shape[-1]):
    plt.subplot(8,8,i+1)
    plt.imshow(f2[0,:,:,i])
    plt.axis('off') 

#third conv
fig3 = plt.figure("Conv 3")
for i in range(f3.shape[-1]):
    plt.subplot(12,12,i+1)
    plt.imshow(f3[0,:,:,i])
    plt.axis('off') 

#fouth conv
fig4 = plt.figure("Conv 4")
for i in range(f4.shape[-1]):
    plt.subplot(12,12,i+1)
    plt.imshow(f4[0,:,:,i])
    plt.axis('off') 
