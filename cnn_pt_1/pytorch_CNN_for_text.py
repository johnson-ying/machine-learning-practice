#try out pytorch code for CNNs for text
#also try out class-based pytorch network

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
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).long()
Ytest = torch.from_numpy(Ytest).long()

#model
class CNNforText(nn.Module):
    def __init__(self):
       super(CNNforText,self).__init__()
       
       self.denseinput = T #dense input - will change as we go through the layers
       
       self.emb = nn.Embedding(maxwords+1, 20)     
       self.conv1 = nn.Sequential(
            nn.Conv1d(20, 32, kernel_size=3),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
       
       self.denseinput = np.ceil(self.denseinput/2)-2  
       
       self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
      
       self.denseinput = np.ceil(self.denseinput/2)-2  
       
       self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
       self.flatten = nn.Flatten()
       
       self.denseinput = np.ceil(self.denseinput/2)-2  
       self.denseinput = int(self.denseinput)
       
       self.dense1 = nn.Sequential(
            nn.Linear(self.denseinput*128, 64),
            nn.ReLU())
       # self.dense2 = nn.Sequential(
       #      nn.Linear(64, 1),  
       #      nn.Sigmoid())
       self.dense2 = nn.Linear(64, 2) #binary problem, but we're just going to use softmax, so output is 2
    
    def forward(self, x):
        out = self.emb(x) #N x T x 20
        out = torch.permute(out, (0,2,1)) #N x 20 x T
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out) #N x 128 x 21
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return out
        # return torch.squeeze(out) #N x batch_size

model = CNNforText()
model.to(device)

loss = torch.nn.CrossEntropyLoss(size_average=True) #binary problem, but we're just going to use softmax
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
    # out = torch.squeeze(out)
    out2 = loss.forward(out, targets) 
    
    out2.backward() 
    optimizer.step()  
    train_loss = out2.item()
    del out
    return train_loss

#get a prediction
def predict(model, inputs):    
    model.eval()
    
    inputs = Variable(inputs, requires_grad = False)

    pred = model.forward(inputs)
    pred = pred.data.cpu().numpy().argmax(axis = 1)
    return pred

#params
B = 64
N = Xtrain.shape[0]
n_batches = int(N//B)
n_iter = 20

#store costs 
train_costs = []

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
