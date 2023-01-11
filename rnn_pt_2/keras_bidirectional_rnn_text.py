
#bidirectional RNN for text data

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional, GlobalMaxPool1D, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd

# https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
#

#multi-label classification
data = pd.read_csv('train.csv')

target = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
inputs = list(data["comment_text"])

max_words = 20000
max_len = 100

#tokenize
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(inputs)
sequences = tokenizer.texts_to_sequences(inputs)

# word2idx
word2idx = tokenizer.word_index

# pad sequences 
data = pad_sequences(sequences, maxlen=max_len)

N, T = data.shape 

#convert data to N x T x D
data = data.reshape((N,T))

n_targ = 6

#model
i = Input(shape = (100,))
emb = Embedding(max_words+1, 50)(i)
# x = LSTM(64, return_sequences = True)(emb)
x = Bidirectional(LSTM(64, return_sequences = True))(emb)
x = GlobalMaxPool1D()(x)
x = Dense(n_targ, activation = 'sigmoid')(x)

model = Model(i, x)

model.compile(
  loss='binary_crossentropy',
  optimizer='Adam',
  metrics=['accuracy'],
)

r = model.fit(data, target, batch_size=64, epochs=5, validation_split=0.2)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')


pred = model.predict(data[0:64,:])
pred[pred>=0.5] = 1
pred[pred<0.5] = 0
true = target[0:64,:]

np.mean(pred == true)
