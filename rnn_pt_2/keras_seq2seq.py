
#seq2seq for english-french translation
#http://www.manythings.org/anki/

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional, GlobalMaxPool1D, Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.optimizers import Adam

#get data
english = []
french = []
for l in open('fra.txt', encoding= "utf-8"):
    values = l.split("\t")
    e = values[0]
    f = values[1]
    english.append(e)
    french.append(f)

#create inputs and target with SOS and EOS tags
inputs = []
targets_input = []
targets_output = []

sample_count = 0
for e,f in zip(english, french):
    inputs.append(e)
    
    t_i = "<sos> " + f
    t_o = f + " <eos>"
    
    targets_input.append(t_i)
    targets_output.append(t_o)
    
    #only train on 12000 sequences b/c of ram issues
    sample_count += 1
    if sample_count > 12000:
        break
    
all_target_sentences = targets_input + targets_output


#tokenize the inputs
max_input_words = 30000
input_tokenizer = Tokenizer(num_words=max_input_words, filters = [])
input_tokenizer.fit_on_texts(inputs)
inputs = input_tokenizer.texts_to_sequences(inputs)

#get input word2idx
input_word2idx = input_tokenizer.word_index
print('# of unique input words: ', len(input_word2idx))

#max length of a sequence
max_len_input = max([len(s) for s in inputs])
print('Max length of input sentence: ', max_len_input)


#tokenize the targets
max_target_words = 60000
target_tokenizer = Tokenizer(num_words=max_target_words, filters = [])
target_tokenizer.fit_on_texts(all_target_sentences)
targets_input = target_tokenizer.texts_to_sequences(targets_input)
targets_output = target_tokenizer.texts_to_sequences(targets_output)

#get input word2idx
target_word2idx = target_tokenizer.word_index
print('# of unique target words: ', len(target_word2idx))

#max length of a sequence
max_len_target = max([len(s) for s in targets_input])
print('Max length of target sentence: ', max_len_target)

assert(target_word2idx['<sos>'])
assert(target_word2idx['<eos>'])



# pad sequences 
X = pad_sequences(inputs, maxlen=max_len_input, padding = "post")
Y_input = pad_sequences(targets_input, maxlen=max_len_target, padding = "post")
Y_target = pad_sequences(targets_output, maxlen=max_len_target, padding = "post")

#total number of target words
v_input = len(input_word2idx)+1

#total number of target words
v_target = len(target_word2idx)+1



#create N x T x vocab_size one hot
Y_one_hot = np.zeros((len(Y_target), max_len_target, v_target))
for n in range(len(Y_target)):
    sentence = Y_target[n,:]
    for t, word in zip(range(len(sentence)), sentence):
        Y_one_hot[n,t,word] += 1




#model
latent_dim = 64

#encoder
enc_i = Input(shape = (max_len_input,))
initial_h = Input(shape = (latent_dim,))
initial_c = Input(shape = (latent_dim,))
enc_emb = Embedding(v_input, 100)
enc_emb_input = enc_emb(enc_i)
enc_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
x, last_h, last_c = enc_lstm(enc_emb_input, initial_state=[initial_h, initial_c])

#decoder
dec_i = Input(shape = (max_len_target,))
dec_emb = Embedding(v_target, 100)
dec_emb_input = dec_emb(dec_i)
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
out, _, _, = dec_lstm(dec_emb_input, initial_state=[last_h, last_c])

dense = Dense(v_target, activation = 'softmax')
x = dense(out)

train_model = Model([enc_i, initial_h, initial_c, dec_i], x)

train_model.summary()



#custom loss to do cross entropy at each time step
def custom_loss(targ, pred):
    mask = K.cast(targ > 0, dtype='float32')
    out = mask * targ * K.log(pred)
    return -K.sum(out) / K.sum(mask)

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



train_model.compile(loss = custom_loss,
              # optimizer = 'Adam',
              optimizer = Adam(learning_rate = 0.005),
              metrics = custom_acc)

z = np.zeros((len(X), latent_dim))
r = train_model.fit([X, z, z, Y_input], Y_one_hot, validation_split = 0.2, batch_size = 64, epochs = 50)

plt.subplot(121)
plt.plot(r.history['val_loss'])
plt.plot(r.history['loss'])
plt.subplot(122)
plt.plot(r.history['val_custom_acc'])
plt.plot(r.history['custom_acc'])



# encoder and decoder model for making predictions 1 word at a time
get_hidden_model = Model([enc_i, initial_h, initial_c], outputs = [last_h, last_c])

get_hidden_model.summary()


predict_input = Input(shape = (1,)) #predict model with only take in 1 word at a time
predict_emb = dec_emb(predict_input)
predict_word, h, c = dec_lstm(predict_emb, initial_state=[initial_h, initial_c]) #during prediction, we'll need output of each hidden state
predict_out = dense(predict_word)

predict_model = Model([predict_input, initial_h, initial_c], [predict_out, h, c])

predict_model.summary()




#now, translate a sentence
input_idx2word = {k:v for v, k in input_word2idx.items()}
input_idx2word[0] = '<eos>'
target_idx2word = {k:v for v, k in target_word2idx.items()}

def original_sentence(input_sentence):
    sentence = ''
    for i in range(input_sentence.shape[1]):
        idx = input_sentence[0,i]
        
        if input_idx2word[idx] == '<eos>':
            break
        sentence = sentence + ' ' + input_idx2word[idx]
    return sentence

def translate_a_sentence(input_sentence):
    z = np.zeros((1, latent_dim))
    h0, c0 = get_hidden_model.predict([input_sentence, z, z], verbose = False)
    
    #feed into decoder model
    input_0 = np.array([[ target_word2idx['<sos>'] ]]) #1 x 1
    word, h, c = predict_model.predict([input_0, h0, c0], verbose=False)
    word = K.argmax(word[0][0]) 
    word = word.numpy()
    
    #add word to sentence
    sentence = '' + target_idx2word[word]
    
    #prepare for next prediction
    word = np.array([[ word ]]) #1x1
    
    
    #keep generating new words until EOS or surpass max length
    for i in range(max_len_target-1):
        word, h, c = predict_model.predict([word, h, c], verbose=False)
        word = K.argmax(word[0][0]) 
        word = word.numpy()
        
        #if EOS, exit
        if target_idx2word[word] == '<eos>':
            break
    
        #add word to sentence
        sentence = sentence + ' ' + target_idx2word[word]
    
        #prepare for next prediction
        word = np.array([[ word ]])
        
    return sentence

#test
print(original_sentence(X[0:1,:]))
print(translate_a_sentence(X[0:1,:]))




#generate 4 lines of poetry
def random_translation():
    idx = np.random.randint(len(X))
    print(original_sentence(X[idx:idx+1,:]))
    print(translate_a_sentence(X[idx:idx+1,:]))


random_translation()
