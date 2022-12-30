
#RNN for text data

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Embedding, Dense, GlobalMaxPooling1D, LSTM, GRU
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split

#load text, get targets
data = pd.read_csv('spam.csv', encoding = "ISO-8859-1") 
data['v1'] = data['v1'].map({'ham':0, 'spam':1})
y = data['v1'].to_numpy()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data['v2'], y, test_size = 0.3)

#tokenize and pad text sequences
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(Xtrain)
Xtrain_tokens = tokenizer.texts_to_sequences(Xtrain)
Xtest_tokens = tokenizer.texts_to_sequences(Xtest)

#zero padding
Xtrain = pad_sequences(Xtrain_tokens, padding = 'post')
T = Xtrain.shape[1]
Xtest = pad_sequences(Xtest_tokens, maxlen = T, padding = 'post')

maxwords = len(tokenizer.word_index)



i = Input(shape = (T,))
embedding = Embedding(maxwords + 1, 20)(i) #embedding dim is 20
x = LSTM(64, return_sequences=True)(embedding) #N x T x 50
x = GlobalMaxPooling1D()(x) #N x T
x = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs = i, outputs = x)

model.compile(
  loss='binary_crossentropy',
  # optimizer = optimizer,
  optimizer='Adam',
  metrics=['accuracy']
)

r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=10, batch_size=64)

# accuracies
plt.plot(r.history['accuracy'], label='train')
plt.plot(r.history['val_accuracy'], label='test')
plt.legend()
