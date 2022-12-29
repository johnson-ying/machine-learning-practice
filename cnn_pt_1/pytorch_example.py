
#try out pytorch code for CNNs for fashion MNIST

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

#Use GPU if possible, if not, then default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

#load data
#reshape to 4D tensor
#in pytorch, it's N x C x H x W
(Xtrain, Ytrain), (Xtest, Ytest) = fashion_mnist.load_data()
Xtrain = np.expand_dims(Xtrain, axis = 1) #N x 1 x H x W
Xtest = np.expand_dims(Xtest, axis = 1) #N x 1 x H x W

#number of classes
K = len(set(Ytrain))

#model
model = nn.Sequential()

model.add_module('conv1', nn.Conv2d(1, 64, (3,3), padding = 'same'))
model.add_module('relu1', nn.ReLU())
model.add_module('bn1', nn.BatchNorm2d(64))
model.add_module('conv2', nn.Conv2d(64, 64, (3,3), padding = 'same'))
model.add_module('bn2', nn.BatchNorm2d(64))
model.add_module('relu2', nn.ReLU())
model.add_module('maxpool1', nn.MaxPool2d((2,2))) #N x 64 x 14 x 14

model.add_module('conv3', nn.Conv2d(64, 128, (3,3), padding = 'same'))
model.add_module('relu3', nn.ReLU())
model.add_module('bn3', nn.BatchNorm2d(128))
model.add_module('conv4', nn.Conv2d(128, 128, (3,3), padding = 'same'))
model.add_module('bn4', nn.BatchNorm2d(128))
model.add_module('relu4', nn.ReLU())
model.add_module('maxpool2', nn.MaxPool2d((2,2))) #N x 128 x 7 x 7

model.add_module('conv5', nn.Conv2d(128, 256, (3,3), padding = 'same'))
model.add_module('relu5', nn.ReLU())
model.add_module('bn5', nn.BatchNorm2d(256))
model.add_module('conv6', nn.Conv2d(256, 256, (3,3), padding = 'same'))
model.add_module('bn6', nn.BatchNorm2d(256))
model.add_module('relu6', nn.ReLU())
model.add_module('maxpool3', nn.MaxPool2d((2,2))) #N x 256 x 3 x 3

model.add_module('flatten', nn.Flatten()) #256 x 3 x 3 = 2304

model.add_module('drop1', nn.Dropout(0.2))
model.add_module('dense1', nn.Linear(256 * 3 * 3, 1024))
model.add_module('d_relu1', nn.ReLU())
model.add_module('drop2', nn.Dropout(0.5))
model.add_module('dense2', nn.Linear(1024, 512))
model.add_module('d_relu2', nn.ReLU())
model.add_module('drop3', nn.Dropout(0.5))
model.add_module('dense3', nn.Linear(512, K))

model.to(device)

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

#convert data to tensors - do not store them in GPU, only store when training
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).long()

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
pred = np.zeros((10000))
B = 100
for j in range(100):
    Xbatch = Xtest[j*B:(j+1)*B].to(device)
    p = predict(model, Xbatch)
    pred[j*B:(j+1)*B] = p

#~93% accuracy
np.mean(pred == Ytest.numpy())
    
