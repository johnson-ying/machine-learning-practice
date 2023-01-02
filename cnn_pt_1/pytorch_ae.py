
#pytorch code for autoencoders 

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

Xtrain = Xtrain/255
Xtest = Xtest/255

#N x D
Xtrain = np.reshape(Xtrain, (len(Xtrain), -1))
Xtest = np.reshape(Xtest, (len(Xtest), -1))

K = len(set(Ytrain))
D = Xtrain.shape[1]

#convert data to tensors - do not store them in GPU, only store when training
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).long()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.encoded = nn.Sequential(nn.Linear(D, 100),
                                     nn.ReLU())
        self.decoded = nn.Sequential(nn.Linear(100, D),
                                     nn.Sigmoid())
    def forward(self, x):
        hidden = self.encoded(x)
        out = self.decoded(hidden)
        return out, hidden

model = RNN()
model.to(device)

#softmax loss
loss = torch.nn.MSELoss(size_average=True) 
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
    
    out, h = model.forward(inputs)
    out = loss.forward(out, targets) 
    
    out.backward() 
    optimizer.step()  
    return out.item()

#get a prediction
def get_hidden(model, inputs):    
    model.eval()
    
    inputs = Variable(inputs, requires_grad = False)

    pred, h = model.forward(inputs)
    return h.cpu().numpy()

#params
B = 200
N = Xtrain.shape[0]
n_batches = int(N//B)
n_iter = 10

#store costs and acc
train_costs = []

#train
for i in range(n_iter):
    train_cost = 0.
    test_cost = 0.
    model.to(torch.device('cuda')) 
    for j in range(n_batches):
        Xbatch = Xtrain[j*B:(j+1)*B].to(device)
        
        train_cost += train(model, loss, optimizer, Xbatch, Xbatch)
    
    print('iter: ', i, 'train cost: ', train_cost / n_batches)        
    train_costs.append(train_cost / n_batches)
    
   
#see recreation
idx = np.random.randint(0,len(Xtrain)-2)
pred, h = model.forward(Xtrain[idx:idx+1,:].to(device))
pred = pred.cpu().detach().numpy()
pred = pred.reshape(28,28)

plt.subplot(121)
plt.imshow(Xtrain[idx:idx+1].reshape(28,28))
plt.gray()

plt.subplot(122)
plt.imshow(pred)
