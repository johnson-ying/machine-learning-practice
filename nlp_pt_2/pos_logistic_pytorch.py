
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
#logistic regresion w/ softmax in pytorch
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

#Use GPU if possible, if not, then default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

Xtrain2 = np.reshape(Xtrain2, (len(Xtrain2), 1, 1))
Xtest2 = np.reshape(Xtest2, (len(Xtest2), 1, 1))

Xtrain = torch.from_numpy(Xtrain2).long()
Ytrain = torch.from_numpy(Ytrain2).long()
Xtest = torch.from_numpy(Xtest2).long()
Ytest = torch.from_numpy(Ytest2).long()

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        
        self.emb = nn.Embedding(total_vocab, total_labels) #N x 1 x total_labels
        
    def forward(self, x):
        out = self.emb(x)
        out = out.reshape(-1,total_labels) #N x 1 x total_labels -> N x total_labels
        return out

model = LogisticRegression()
model.to(device)

#binary cross entropy loss
loss = torch.nn.CrossEntropyLoss(size_average=True)
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

#get a loss without training
def get_loss(model, loss, inputs, targets):
    model.eval()
    
    inputs = Variable(inputs, requires_grad = False)
    inputs.to(device)
    targets = Variable(targets, requires_grad = False)
    targets.to(device)
    
    out = model.forward(inputs)
    out = loss.forward(out, targets) 
    return out.item()

#get a prediction
def predict(model, inputs):    
    model.eval()
    
    inputs = Variable(inputs, requires_grad = False)

    pred = model.forward(inputs)
    pred = pred.data.cpu().numpy().argmax(axis = 1)
    return pred

#get a prediction
def get_accuracy(model, inputs, targets):    
    p = predict(model, inputs)
    return np.round(np.mean(p == targets.cpu().numpy()), 2)

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
        
        train_cost += train(model, loss, optimizer, Xbatch, Ybatch)
    
    test_cost = get_loss(model, loss, Xtest.to(device), Ytest.to(device))    
    
    train_acc = get_accuracy(model, Xtrain.to(device), Ytrain.to(device))
    test_acc = get_accuracy(model, Xtest.to(device), Ytest.to(device))
    
    print('iter: ', i,
          'train cost: ', train_cost / n_batches,
          'test cost: ', test_cost,
          'train accuracy: ', train_acc,
          'test accuracy: ', test_acc)    
    
    train_costs.append(train_cost / n_batches)
    test_costs.append(test_cost)
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    
    
pred = predict(model, Xtest.to(device))
from sklearn.metrics import f1_score

f1_score(Ytest2, pred, average=None).mean()

