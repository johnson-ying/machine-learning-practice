
#try out pytorch code for RNNs for text

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split

#Use GPU if possible, if not, then default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

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


#convert data to tensors - do not store them in GPU, only store when training
Xtrain = torch.from_numpy(Xtrain).long()
Ytrain = torch.from_numpy(Ytrain).float()
Xtest = torch.from_numpy(Xtest).long()
Ytest = torch.from_numpy(Ytest).float()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.emb = nn.Embedding(maxwords + 1, 20)
        self.lstm = nn.LSTM(20, 64)
        self.maxpool = nn.MaxPool1d(64)
        self.dense = nn.Linear(T, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        out = self.emb(x)
        out, (h,c) = self.lstm(out)
        out = self.maxpool(out)
        out = out.reshape(-1,T)
        out = self.dense(out)
        out = self.sig(out) #sigmoid for binary cross entropy
        return torch.squeeze(out) #get 1D output

model = RNN()
model.to(device)

#binary cross entropy loss
loss = torch.nn.BCELoss(size_average=True) 
loss.to(device)

optimizer = optim.Adam(model.parameters())

#train, and get train cost
def train(model, loss, optimizer, inputs, targets):
    model.train()
    
    inputs = Variable(inputs, requires_grad = False)
    inputs.to(device)
    targets = Variable(targets, requires_grad = False)
    targets.to(device)

    optimizer.zero_grad()
    
    out = model.forward(inputs)
    out = loss.forward(out, targets) 
    
    out.backward() 
    optimizer.step()  
    return out.item()

#get a prediction
def predict(model, inputs):    
    model.eval()
    
    inputs = Variable(inputs, requires_grad = False)

    pred = model.forward(inputs)
    pred = torch.round(pred)
    pred = pred.data.cpu().numpy()
    return pred

#params
B = 64
N = Xtrain.shape[0]
n_batches = int(N//B)
n_iter = 20

#store costs and acc
train_costs = []
test_costs = []
train_accs = []
test_accs = []

#train
for i in range(n_iter):
    train_cost = 0.
    test_cost = 0.
    model.to(torch.device('cuda')) 
    for j in range(n_batches):
        Xbatch = Xtrain[j*B:(j+1)*B].to(device)
        Ybatch = Ytrain[j*B:(j+1)*B].to(device)
        
        train_cost += train(model, loss, optimizer, Xbatch, Ybatch)
    
    print('iter: ', i, 'train cost: ', train_cost / n_batches)        
    train_costs.append(train_cost / n_batches)
    
   
#grab predictions batch by batch so dont overload memory
pred = predict(model, Xtest.to(device))

#~98% accuracy
np.mean(pred == Ytest.numpy())
    
