
#try out pytorch code for RNNs for images

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from keras.datasets import fashion_mnist

#Use GPU if possible, if not, then default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)



#data already in shape N x T x D
(Xtrain, Ytrain), (Xtest, Ytest) = fashion_mnist.load_data()

K = len(set(Ytrain))
T, D = Xtrain.shape[1], Xtrain.shape[2]



#convert data to tensors - do not store them in GPU, only store when training
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).long()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.lstm = nn.LSTM(D, 100, batch_first=True)
        self.dense = nn.Linear(100, K)
        
    def forward(self, x):
        out, (h,c) = self.lstm(x)
        h = h[0,:,:] #get final hidden state
        out = self.dense(h)
        return out

model = RNN()
model.to(device)

#softmax loss
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
n_iter = 10

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

#~83% accuracy
np.mean(pred == Ytest.numpy())
    
