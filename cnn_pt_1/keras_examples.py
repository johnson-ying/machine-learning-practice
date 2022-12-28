
#try out diff keras CNN architectures for fashion MNIST

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

# i = Input(shape = Xtrain[0].shape)
# x = Conv2D(32, (3,3), strides = 2, activation = 'relu', padding = 'same')(i)
# x = Conv2D(64, (3,3), strides = 2, activation = 'relu', padding = 'same')(x)
# x = Conv2D(128, (3,3), strides = 2, activation = 'relu', padding = 'same')(x)
# x = Flatten()(x)
# x = Dense(500, activation = 'relu')(x)
# x = Dense(K, activation = 'softmax')(x)


# i = Input(shape = Xtrain[0].shape)
# x = Conv2D(32, (3,3), strides = 2, activation = 'relu', padding = 'same')(i)
# x = BatchNormalization()(x)
# # x = Dropout(0.25)(x)
# x = Conv2D(64, (3,3), strides = 2, activation = 'relu', padding = 'same')(x)
# x = BatchNormalization()(x)
# # x = Dropout(0.25)(x)
# x = Conv2D(128, (3,3), strides = 2, activation = 'relu', padding = 'same')(x)
# x = BatchNormalization()(x)
# # x = Dropout(0.25)(x)
# x = Flatten()(x)
# x = Dense(500, activation = 'relu')(x)
# x = Dense(K, activation = 'softmax')(x)

#~93% accuracy
i = Input(shape = Xtrain[0].shape)
x = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(i)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(K, activation = 'softmax')(x)


model = Model(inputs = i, outputs = x)

# optimizer = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999)

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

###
###
### try improving results with image generator

from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    shear_range = 0.1,
                                    brightness_range = [0.7,1.3])

train_generator = data_generator.flow(Xtrain, Ytrain, batch_size = 64)

# # ~91% accuracy
# i = Input(shape = Xtrain[0].shape)
# x = Conv2D(32, (3,3), strides = 2, activation = 'relu', padding = 'same')(i)
# x = BatchNormalization()(x)
# x = Conv2D(64, (3,3), strides = 2, activation = 'relu', padding = 'same')(x)
# x = BatchNormalization()(x)
# x = Conv2D(128, (3,3), strides = 2, activation = 'relu', padding = 'same')(x)
# x = BatchNormalization()(x)
# x = Flatten()(x)
# x = Dense(500, activation = 'relu')(x)
# x = Dense(K, activation = 'softmax')(x)

# ~93 accuracy
i = Input(shape = Xtrain[0].shape)
x = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(i)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
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
  optimizer='Adam',
  metrics=['accuracy']
)

r = model.fit_generator(train_generator, steps_per_epoch = Xtrain.shape[0]//64, epochs = 15)

# accuracies
plt.plot(r.history['accuracy'], label='train')
plt.plot(r.history['val_accuracy'], label='test')
plt.legend()

pred = model.predict(Xtest)
pred = np.argmax(pred, axis = 1)
print(np.mean(pred == Ytest)) 
