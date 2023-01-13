
#attention for english-french translation
#http://www.manythings.org/anki/

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional, GlobalMaxPool1D, Embedding, RepeatVector, Concatenate, Dot, Reshape, Lambda
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

# skip the really easy sentences
english = english[100000:]
french = french[100000:]

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
    
    #only train on 8000 sequences b/c of ram issues
    sample_count += 1
    if sample_count > 8000:
        break
    
all_target_sentences = targets_input + targets_output


#tokenize the inputs
max_input_words = 30000
input_tokenizer = Tokenizer(num_words=max_input_words, filters = '.,?!')
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
target_tokenizer = Tokenizer(num_words=max_target_words, filters = '.,?!')
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
latent_dim = 400

# #softmax over time activation function
# def softmax_over_time(y):
#     #y is N x T x 1
#     #take softmax over T dimension
#     return K.exp(y) / K.sum(K.exp(y), axis = 1, keepdims = True)
#     #return output is same shape 


def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s



#encoder
enc_i = Input(shape = (max_len_input,)) #N x T x 1
enc_emb = Embedding(v_input, 100) 
enc_emb_input = enc_emb(enc_i) #N x T x 100
enc_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True))
enc_states = enc_lstm(enc_emb_input) #N x T x latent_dim*2

#decoder
dec_i = Input(shape = (max_len_target,)) #N x T' x 1
dec_emb = Embedding(v_target, 100) 
dec_emb_input = dec_emb(dec_i) #N x T x 100

#attention class - get context vector
attn_repeat = RepeatVector(max_len_input) #replicates the decoders prev. hidden state to same amount of encoder hidden states 
attn_concat = Concatenate(axis=-1) #concatenates the replicated decoder states to the encoder hidden states
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes = 1)

def attention(h, s_1):
    #h is all hidden states of encoder
    
    #repeat the hidden states
    s_1_repeat = attn_repeat(s_1) #N x latent_dim -> N x max_len_input x latent_dim
    
    #concat all hidden states to all encoder hidden states
    h_s_concat = attn_concat([h, s_1_repeat]) #N x max_len_input x (2*latent_dim + latent_dim)

    out = attn_dense1(h_s_concat) #N x max_len_input x 10
    alphas = attn_dense2(out)

    #dot the alphas with all hidden states
    #alphas is N x T x 1
    #hidden states of encoder is N x T x latent_dim
    context = attn_dot([alphas, h]) #N x 1 x latent_dim
    return context, alphas #also return alphas if you want to plot

#continue decoder architecture
initial_s = Input(shape = (latent_dim,))
initial_c = Input(shape = (latent_dim,))
dec_lstm = LSTM(latent_dim, return_state=True)
dec_dense = Dense(v_target, activation = 'softmax')

concat_layer = Concatenate(axis = 2)

s = initial_s
c = initial_c

outputs = []
alphas = []
for t in range(max_len_target):
    #do a step of attention
    context, alphs = attention(enc_states, s)
    
    #store alphas if you want
    alphas.append(alphs)
    
    #get previous word
    selector = Lambda(lambda x: x[:, t:t+1])
    prev_word = selector(dec_emb_input) #N x 1 x 50
    
    #concat prev. word to context vector
    context_prev_word = concat_layer([context, prev_word]) #N x 1 x (2*latent_dim + 100)

    #feed into lstm
    out, s, c = dec_lstm(context_prev_word, initial_state = [s, c]) #each output is N x latent_dim

    #predict next word
    out = dec_dense(out) #N x vocab size 

    outputs.append(out) #a list of N x vocab_size

#targets are N x T x vocab_size
#need to change the list output into a tensor of N x T x vocab size

#first, convert the list output into tensor of T x N x vocab size
stacker = K.stack(outputs) 

#then, permute dim to N x T x vocab size
permute = K.permute_dimensions(stacker, (1,0,2))



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


train_model = Model(inputs = [enc_i, dec_i, initial_s, initial_c], 
                    outputs = permute)


train_model.summary()

train_model.compile(loss = custom_loss,
              # optimizer = Adam(learning_rate = 0.01),
              optimizer = 'Adam',
              metrics = custom_acc)

z = np.zeros((len(X), latent_dim))
r = train_model.fit([X, Y_input, z, z], Y_one_hot, validation_split = 0.1, batch_size = 64, epochs = 70)

plt.subplot(121)
plt.plot(r.history['val_loss'])
plt.plot(r.history['loss'])
plt.subplot(122)
plt.plot(r.history['val_custom_acc'])
plt.plot(r.history['custom_acc'])


#time to make predictions

#model to get all of encoder hidden states 
get_enc_states = Model(enc_i, enc_states)

#decoder
enc_states_input = Input(shape = (max_len_input, latent_dim*2,)) #N x T x latent_dim*2
single_dec_input = Input(shape = (1,)) #single decoder word
single_dec_input_emb = dec_emb(single_dec_input) 

#context vector
context_vector, alphss = attention(enc_states_input, initial_s) 

#concat context vector to previously decoded word
context_w_word = concat_layer([context_vector, single_dec_input_emb])

single_pred, ss, cc = dec_lstm(context_w_word, initial_state = [initial_s, initial_c])

single_pred_word = dec_dense(single_pred)

pred_model = Model(inputs = [enc_states_input, single_dec_input, initial_s, initial_c],
                   outputs = [alphss, single_pred_word, ss, cc])



input_idx2word = {k:v for v, k in input_word2idx.items()}
input_idx2word[0] = '<eos>'
target_idx2word = {k:v for v, k in target_word2idx.items()}


def translate_a_sentence(input_sentence, s0, c0):
    enc_states = get_enc_states(input_sentence)
    allalphas = []
    
    input_0 = np.array([[ target_word2idx['<sos>'] ]]) #1 x 1
    
    a, word, s, c = pred_model.predict([enc_states, input_0, s0, c0], verbose = False)
    
    word = K.argmax(word[0]) 
    word = word.numpy()
    
    #add word to sentence
    sentence = '' + target_idx2word[word]
    
    #prepare for next prediction
    word = np.array([[ word ]]) #1x1
    
    #store alphas
    allalphas.append(a)
    
    #keep translating until max length or eos
    for t in range(max_len_target-1):
        
        a, word, s, c = pred_model.predict([enc_states, word, s, c], verbose = False)
        
        word = K.argmax(word[0]) 
        word = word.numpy()
        
        #if got a zero, break
        if word == 0:
            break
        
        #if EOS, exit
        if target_idx2word[word] == '<eos>':
            break
        
        sentence = sentence + ' ' + target_idx2word[word]
        
        allalphas.append(a)
        
        word = np.array([[ word ]]) #1x1
    
    return sentence, allalphas
   
def original_sentence(input_sentence):
    sentence = ''
    for i in range(input_sentence.shape[1]):
        idx = input_sentence[0,i]
        
        if input_idx2word[idx] == '<eos>':
            break
        sentence = sentence + ' ' + input_idx2word[idx]
    return sentence
 
    
    
#test
print(original_sentence(X[0:1,:]))

zz = np.zeros((1, latent_dim))
translated, al = translate_a_sentence(X[0:1,:], zz, zz)
print(translated)



#translate a random sentence
def random_translation():
    idx = np.random.randint(len(X))
    print(original_sentence(X[idx:idx+1,:]))
    zz = np.zeros((1, latent_dim))
    translated, al = translate_a_sentence(X[idx:idx+1,:], zz, zz)
    print(translated)


random_translation()

