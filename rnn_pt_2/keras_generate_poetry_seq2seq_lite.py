
#NOT seq2seq implementation. coding practice before implementing seq2seq
#generate poetry using seq2seq concepts

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional, GlobalMaxPool1D, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.optimizers import Adam

x = []
for l in open('robert_frost.txt'):
    x.append(l)

x = [l for l in x if len(l) > 2]

#create inputs and target with SOS and EOS tags
inputs = []
targets = []
for l in x:
    i = "<sos> " + l
    t = l + " <eos>"
    
    inputs.append(i)
    targets.append(t)

allsentences = inputs + targets

max_words = 20000

#tokenize
tokenizer = Tokenizer(num_words=max_words, filters = '.,')
tokenizer.fit_on_texts(allsentences)
inputs = tokenizer.texts_to_sequences(inputs)
targets = tokenizer.texts_to_sequences(targets)

#max length of a sequence
max_len = max([len(s) for s in inputs])

# word2idx
word2idx = tokenizer.word_index
assert(word2idx['<sos>'])
assert(word2idx['<eos>'])

# pad sequences 
X = pad_sequences(inputs, maxlen=max_len, padding = "post")
Y = pad_sequences(targets, maxlen=max_len, padding = "post")

#total number of words
v = len(word2idx)+1

#create N x T x D targets
Y_one_hot = np.zeros((len(Y), max_len, v))
for n in range(len(Y)):
    sentence = Y[n,:]
    for t, word in zip(range(len(sentence)), sentence):
        Y_one_hot[n,t,word] += 1

#model
latent_dim = 64

i = Input(shape = (max_len,))
initial_h = Input(shape = (latent_dim,))
initial_c = Input(shape = (latent_dim,))
emb = Embedding(max_len + 1, 100)
emb_input = emb(i)
lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
x, _, _ = lstm(emb_input, initial_state=[initial_h, initial_c])
dense = Dense(v, activation = 'softmax')
x = dense(x)

train_model = Model([i, initial_h, initial_c], x)

train_model.summary()

train_model.compile(loss = 'categorical_crossentropy',
              # optimizer = 'Adam',
              optimizer = Adam(learning_rate = 0.005),
              metrics = ['accuracy'])

z = np.zeros((len(X), latent_dim))
r = train_model.fit([X, z, z], Y_one_hot, validation_split = 0.2, batch_size = 64, epochs = 150)

plt.plot(r.history['val_accuracy'])
plt.plot(r.history['accuracy'])


# model for making predictions 1 word at a time
predict_input = Input(shape = (1,)) #predict model with only take in 1 word at a time
predict_emb = emb(predict_input)
predict_word, h, c = lstm(predict_emb, initial_state=[initial_h, initial_c]) #during prediction, we'll need output of each hidden state
predict_out = dense(predict_word)

predict_model = Model([predict_input, initial_h, initial_c], [predict_out, h, c])

predict_model.summary()




#now, predict a sentence
idx2word = {k:v for v, k in word2idx.items()}

input_0 = np.array([[ word2idx['<sos>'] ]]) #1 x 1
h0 = np.zeros((1, latent_dim)) #1 x latent_dim
c0 = np.zeros((1, latent_dim)) #1 x latent dim


def predict_a_sentence(input_0, h0, c0):
    
    word, h, c = predict_model.predict([input_0, h0, c0], verbose=False)
    word = word[0][0] #the softmax probabilities
    
    #randomly sample a word - as long as its not sos or eos
    while True:
        word_idx = np.random.choice(v, p = word)
        if idx2word[word_idx] != '<eos>' and idx2word[word_idx] != '<sos>' and word_idx != 0:
            break
    
    #add word to sentence
    sentence = '' + idx2word[word_idx]
    
    #prepare for next prediction
    word = np.array([[ word_idx ]])
    
    #keep generating new words until EOS or surpass max length
    for i in range(max_len-1):
        word, h, c = predict_model.predict([word, h, c], verbose=False)
        word = word[0][0]
        
        #if EOS, exit
        if idx2word[np.argmax(word)] == '<eos>':
            break
        
        #randomly sample a word - as long as its not sos or eos
        while True:
            word_idx = np.random.choice(v, p = word)
            if idx2word[word_idx] != '<eos>' and idx2word[word_idx] != '<sos>' and word_idx != 0:
                break
        
        #add word to sentence
        sentence = sentence + ' ' + idx2word[word_idx]
    
        #prepare for next prediction
        word = np.array([[ word_idx ]])
        
    return sentence

print(predict_a_sentence(input_0, h0, c0))

#generate 4 lines of poetry
def generate_4_lines():
    print('')
    for i in range(4):
        print(predict_a_sentence(input_0, h0, c0))



generate_4_lines()
