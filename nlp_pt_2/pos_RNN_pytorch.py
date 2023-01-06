
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
#RNN in pytorch
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

#Use GPU if possible, if not, then default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

Xtrain = torch.from_numpy(Xtrain).long()
Ytrain = torch.from_numpy(Ytrain2).long()
Xtest = torch.from_numpy(Xtest).long()
Ytest = torch.from_numpy(Ytest2).long()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.emb = nn.Embedding(total_vocab, 20)
        self.lstm = nn.LSTM(20, 64)
        self.timedist_dense = nn.Linear(64, total_labels) 
        self.softmax = nn.Softmax(dim = -1)
        #apparently pytorch linear layer can generalize to 3D inputs, analogous to TimeDistr. in keras
        
    def forward(self, x):
        out = self.emb(x) #N x T x 20
        out, (h,c) = self.lstm(out) #N x T x 64
        out = self.timedist_dense(out) #N x T x total_labels
        return self.softmax(out) #N x T x total_labels

model = RNN()
model.to(device)

#custom loss to do cross entropy at all time steps 
def my_loss(targ, pred):
    target_mask = targ > 0 #pick out target indexes 

    out = target_mask * targ * torch.log(pred)
    loss = -torch.sum(out) / torch.sum(target_mask)
    return loss

y = model.forward(Xtrain[0:64].to(device))

optimizer = optim.Adam(model.parameters())

#train, and get train cost
def train(model, optimizer, inputs, targets):
    model.train()
    
    inputs = Variable(inputs, requires_grad = False)
    inputs.to(device)
    targets = Variable(targets, requires_grad = False)
    targets.to(device)

    optimizer.zero_grad()
    
    out = model.forward(inputs)
    
    #loss defined in here
    out = my_loss(targets, out).to(device)
    
    out.backward() 
    optimizer.step()  
    return out.item()

#get a loss without training
def get_loss(model, inputs, targets):
    model.eval()
    
    inputs = Variable(inputs, requires_grad = False)
    inputs.to(device)
    targets = Variable(targets, requires_grad = False)
    targets.to(device)
    
    out = model.forward(inputs)
    #loss
    out = my_loss(targets, out).to(device)
    return out.item()

#get a prediction
def predict(model, inputs):    
    model.eval()
    inputs = Variable(inputs, requires_grad = False)
    pred = model.forward(inputs)
    return pred

#custom accuracy
def custom_acc(model, inputs, targ):
    pred = predict(model, inputs)  
    # target and pred are both size N x T x total_labels
    # take the argmax along the 3rd dimension
  
    targ = torch.argmax(targ, axis=-1) #N x T
    pred = torch.argmax(pred, axis=-1) #N x T
  
    correct = targ == pred #N x T

    # discount all the 0 padded values
    target_mask = targ > 0 #pick out target indexes N x T
    n_correct = torch.sum(target_mask * correct)
    n_total = torch.sum(target_mask)
  
    acc = n_correct / n_total
    return acc.cpu().numpy()


#params
B = 200
N = Xtrain.shape[0]
n_batches = int(N//B)
n_iter = 40

#store costs and acc
train_costs = []
test_costs = []
train_accs = []
test_accs = []

#train
for i in range(n_iter):
    train_cost = 0.
    
    for j in range(n_batches):
        Xbatch = Xtrain[j*B:(j+1)*B].to(device)
        Ybatch = Ytrain[j*B:(j+1)*B].to(device)
        
        train_cost += train(model, optimizer, Xbatch, Ybatch)
    
    test_cost = get_loss(model, Xtest.to(device), Ytest.to(device))    

    train_acc = custom_acc(model, Xtrain.to(device), Ytrain.to(device))
    test_acc = custom_acc(model, Xtest.to(device), Ytest.to(device))
    
    print('iter: ', i,
          'train cost: ', train_cost / n_batches,
           'test cost: ', test_cost,
            'train accuracy: ', train_acc,
            'test accuracy: ', test_acc)    
    
    train_costs.append(train_cost / n_batches)
    test_costs.append(test_cost)
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    
#
#
# Test out on a random sequence
idx = np.random.randint(len(Xtrain))
sequence = Xtrain[idx:idx+1].numpy()

idx2word = {v:k for k, v in word2idx.items()}
sentence = ''
for i in sequence[0]:
    #break if encountered a zero pad value
    if i == 0:
        break
    sentence += idx2word[i] + ' '

print(sentence)

#get pos preds
input_tensor = torch.from_numpy(sequence).long()
pos_preds = predict(model, input_tensor.to(device))
pos_preds = pos_preds.cpu().detach().numpy()
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
