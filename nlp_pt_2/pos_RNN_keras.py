
#parts of speech classification using RNN in keras and pytorch
#input is a sequence of time steps, predict pos tag for each time step
# https://www.clips.uantwerpen.be/conll2000/chunking/

import numpy as np
import matplotlib.pyplot as plt
import string

def tokenize(s):
    s = s.lower()
    # s = s.translate(str.maketrans('', '', string.punctuation)) #dont want to remove punctuations in this case
    words = s.split()
    return words

Xtrain = []
Xtest = []
Ytrain = []
Ytest = []

#since we'll be looking at sequences of diff length, we'll have to zero-pad sequences to same length
#which means 0 should never be counted as a "real" input or "target"

word2idx = {'zero_pad':0} #0 is specific to zero padding
word2idx['UNK'] = 1 #add an unknown key for test words not encountered during training
total_vocab = 2

target2idx = {'zero_pad':0} #0 is specific to zero padding
target2idx['UNK'] = 1 #add an extra unknown label for test labels not encountered during training
total_labels = 2

#as we extract sentences, we will also fill out word2idx and target2idx
x = []
y = []
for line in open('train.txt'):
    if len(line)>2: #make sure it isnt empty, otherwise its the end of the sentence
        words = tokenize(line)
        
        #add word to word2idx
        if words[0] not in word2idx.keys():
            word2idx[words[0]] = total_vocab
            total_vocab += 1
        
        #add label to target2idx
        if words[1] not in target2idx.keys():
            target2idx[words[1]] = total_labels
            total_labels += 1
        
        x.append(word2idx[words[0]])
        y.append(target2idx[words[1]])
        
    else: #if line was empty, it means the sentence ended. append whole sentence. 
        Xtrain.append(x)
        Ytrain.append(y)
        x = [] #reset
        y = [] #reset
        
#do the same for test data 
x = []
y = []
for line in open('test.txt'):
    if len(line)>2: #make sure it isnt empty, otherwise its the end of the sentence
        words = tokenize(line)
        
        word = words[0]
        label = words[1] 
        
        #add word to word2idx
        if word not in word2idx.keys():
            word = 'UNK'
        
        #add label to target2idx
        if label not in target2idx.keys():
            label = 'UNK'
        
        x.append(word2idx[word])
        y.append(target2idx[label])
        
    else: #if line was empty, it means the sentence ended. append whole sentence. 
        Xtest.append(x)
        Ytest.append(y)
        x = [] #reset
        y = [] #reset    
    
#zero pad sequences to same length
max_length = max([len(t) for t in Xtrain])    
# max_length = max([len(t) for t in Xtest])    

from keras.utils import pad_sequences
Xtrain = pad_sequences(Xtrain, padding = 'post')
T = Xtrain.shape[1]
Xtest = pad_sequences(Xtest, maxlen = T, padding = 'post')
    
#inputs are N x T
    
# create targets, cannot use sparse categorical cross entropy with sequences
# need to change targets into form N x T x total_labels 
# 1 true target for each time step

Ytrain2 = np.zeros((len(Xtrain), T, total_labels), dtype='float32')
Ytest2 = np.zeros((len(Xtest), T, total_labels), dtype='float32')

# assign targets
for n, l in zip(range(len(Ytrain)), Ytrain):
    for t, word in zip(range(len(l)), l):
        Ytrain2[n, t, word] = 1

for n, l in zip(range(len(Ytest)), Ytest):
    for t, word in zip(range(len(l)), l):
        Ytest2[n, t, word] = 1


#
#RNN in keras
import keras
import keras.backend as K
from keras.layers import Dense, Input, Embedding, LSTM, TimeDistributed
from keras.models import Model

i = Input(shape = (T,)) #N x T
emb = Embedding(total_vocab, 20)(i) #N x T X 20
lstm = LSTM(64, return_sequences = True)(emb) #N x T x 64

#create a dense + softmax for each hidden state, to predict target
timedist = TimeDistributed(Dense(total_labels, activation = 'softmax'))(lstm) #N x T x total_labels

model = Model(inputs = i, outputs = timedist)

#custom loss to do cross entropy at each time step
def custom_loss(targ, pred):
    # target and pred are both size N x T x total_labels
    target_mask = K.cast(targ > 0, dtype='float32')
  
    #cross entropy at each time step
    out = target_mask * targ * K.log(pred) #N x T x total_labels
    return -K.sum(out) / K.sum(target_mask) #return the mean cost

def custom_acc(targ, pred):
    # target and pred are both size N x T x total_labels
    # take the argmax along the 3rd dimension
    targ = K.argmax(targ, axis=-1) #N x T
    pred = K.argmax(pred, axis=-1) #N x T
    correct = K.cast(K.equal(targ, pred), dtype='float32') #N x T

    # discount all the 0 padded values
    target_mask = K.cast(targ > 0, dtype='float32') # N x T
    n_correct = K.sum(target_mask * correct)
    n_total = K.sum(target_mask)
    return n_correct / n_total

model.compile(optimizer = 'adam',
              loss = custom_loss,
              metrics = [custom_acc])

r = model.fit(Xtrain, Ytrain2, batch_size = 200, validation_data = (Xtest, Ytest2), epochs = 20)

plt.subplot(121)
plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.subplot(122)
plt.plot(r.history['custom_acc'])
plt.plot(r.history['val_custom_acc'])

#
#
# Test out on a random sequence
idx = np.random.randint(len(Xtrain))
sequence = Xtrain[idx:idx+1]

idx2word = {v:k for k, v in word2idx.items()}
sentence = ''
for i in sequence[0]:
    #break if encountered a zero pad value
    if i == 0:
        break
    sentence += idx2word[i] + ' '

print(sentence)

#get pos preds
pos_preds = model.predict(sequence)
pos_preds = pos_preds[0]
pos_preds = np.argmax(pos_preds, axis = 1)

idx2label = {v:k for k, v in target2idx.items()}

tags = ''
for i in pos_preds:
    tags += idx2label[i] + ' '
    #break if encountered a period
    if idx2label[i] == '.':
        break
    
print(sentence)
print(tags)
