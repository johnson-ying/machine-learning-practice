
#parts of speech classification using logistic regression w/ softmax in keras and pytorch
#each word is a separate input, its tag is a separate output
# https://www.clips.uantwerpen.be/conll2000/chunking/

import numpy as np
import matplotlib.pyplot as plt
import string

#loading in data..
train = []
for l in open('train.txt'):
    train.append(l)
train = [t for t in train if len(t) > 2]
    
test = []
for l in open('test.txt'):
    test.append(l)
test = [t for t in test if len(t) > 2] 

def tokenize(s):
    s = s.lower()
    # s = s.translate(str.maketrans('', '', string.punctuation)) #dont want to remove punctuations in this case
    words = s.split()
    return words

Xtrain = []
Ytrain = []
for l in train:
    words = tokenize(l)
    Xtrain.append(words[0])
    Ytrain.append(words[1])
 
Xtest = []
Ytest = []
for l in test:
    words = tokenize(l)
    Xtest.append(words[0])
    Ytest.append(words[1])
 
#word2idx
word2idx = {'UNK':0} #add an unknown key for test words not encountered during training
total_vocab = 1

for w in Xtrain:
    if w not in word2idx.keys():
        word2idx[w] = total_vocab
        total_vocab += 1

#target2idx - convert each target label to an idx
K = len(set(Ytrain)) #44 different labels
target2idx = {'UNK':0} #add an extra unknown label for test labels not encountered during training
total_labels = 1

for w in Ytrain:
    if w not in target2idx.keys():
        target2idx[w] = total_labels
        total_labels += 1

#convert train and test to their idx form
# Xtrain2 = np.zeros((len(Xtrain), total_vocab)) #<- cant do this, file would be way too large, just store as an array of idx
Xtrain2 = np.zeros((len(Xtrain)))
for n in range(len(Xtrain)):
        Xtrain2[n] = word2idx[Xtrain[n]]
Xtrain2 = Xtrain2.astype(int)

Ytrain2 = np.zeros((len(Ytrain)))
for n in range(len(Ytrain)):
        Ytrain2[n] = target2idx[Ytrain[n]]
Ytrain2 = Ytrain2.astype(int)

Xtest2 = np.zeros((len(Xtest)))
for n in range(len(Xtest)):
    word = Xtest[n]
    if word not in word2idx.keys():
        word = 'UNK'
    Xtest2[n] = word2idx[word]
Xtest2 = Xtest2.astype(int) 

Ytest2 = np.zeros((len(Ytest)))
for n in range(len(Ytest)):
    label = Ytest[n]
    if label not in target2idx.keys():
        label = 'UNK'
    Ytest2[n] = target2idx[label]
Ytest2 = Ytest2.astype(int) 

#
#logistic regresion w/ softmax in keras
import keras
import keras.backend as K
from keras.layers import Dense, Input, Embedding
from keras.models import Model

Xtrain2 = np.reshape(Xtrain2, (len(Xtrain2), 1, 1))
Xtest2 = np.reshape(Xtest2, (len(Xtest2), 1, 1))

i = Input(shape = (1,1)) #N x 1 x 1
emb = Embedding(total_vocab, total_labels)(i) #N x 1 X total_labels
reshape = K.reshape(emb, (-1, total_labels)) #reshape to N x total_labels
softmax = keras.layers.Softmax(axis=-1) #take softmax
x = softmax(reshape)

model = Model(inputs = i, outputs = x)

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = 'accuracy')

r = model.fit(Xtrain2, Ytrain2, batch_size = 200, validation_data = (Xtest2, Ytest2), epochs = 10)

plt.subplot(121)
plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.subplot(122)
plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])


pred = model.predict(Xtest2)
pred = np.argmax(pred, axis = 1)
from sklearn.metrics import f1_score

f1_score(Ytest2, pred, average=None).mean()
