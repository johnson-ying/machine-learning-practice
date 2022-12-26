
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import optim
from keras.datasets import mnist


#Use GPU if possible, if not, then default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


# load MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# standardize data between 0 and 1
test_X = test_X / 255
train_X = train_X / 255

# resize to N x D 
train_X = np.resize(train_X, (60000,784))
test_X = np.resize(test_X, (10000,784))


N, D = train_X.shape
K = len(set(train_y))



#model
model = torch.nn.Sequential()

model.add_module('dense1', torch.nn.Linear(D, 300))
model.add_module('relu1', torch.nn.ReLU())
model.add_module('dense2', torch.nn.Linear(300, 100))
model.add_module('relu2', torch.nn.ReLU())
model.add_module('dense3', torch.nn.Linear(100, K))
model.to(device)

loss = torch.nn.CrossEntropyLoss(size_average=True)
loss.to(device)

optimizer = optim.Adagrad(model.parameters())

#train and get train cost
def train(model, loss, optimizer, inputs, targets): 
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

#get test cost
def get_test_cost(model, inputs, targets): 
    inputs = Variable(inputs)
    inputs.to(device)
    
    targets = Variable(targets) 
    targets.to(device)
    
    out = model.forward(inputs)
    out = loss.forward(out, targets) 
    return out.item()

#get a prediction
def predict(model, inputs):  
    inputs = Variable(inputs, requires_grad = False)
    inputs.to(device)
    
    pred = model.forward(inputs)
    pred = pred.data.cpu().numpy().argmax(axis = 1)
    return pred

#convert data to tensors
Xtrain = torch.from_numpy(train_X).float().to(device)
Ytrain = torch.from_numpy(train_y).long().to(device)
Xtest = torch.from_numpy(test_X).float().to(device)
Ytest = torch.from_numpy(test_y).long().to(device)

#params
B = 512
n_batches = int(N//B)
n_iter = 10

#store costs and acc
train_costs = []
test_costs = []
train_accs = []
test_accs = []

#train
for i in range(n_iter):
    print('iter: ', i)
    train_cost = 0.
    test_cost = 0.
    for j in range(n_batches):
        Xbatch = Xtrain[j*B:(j+1)*B]
        Ybatch = Ytrain[j*B:(j+1)*B]
        
        train_cost += train(model, loss, optimizer, Xbatch, Ybatch)
        test_cost += get_test_cost(model, Xtest, Ytest)
        
    train_costs.append(train_cost / n_batches)
    test_costs.append(test_cost / n_batches)
    
    #also get accuracies
    train_pred = predict(model, Xtrain)
    test_pred = predict(model, Xtest)
    
    train_accs.append(np.mean(train_pred == Ytrain.cpu().numpy())) 
    test_accs.append(np.mean(test_pred == Ytest.cpu().numpy())) 
    

#plot
plt.plot(train_costs)
plt.plot(test_costs)


plt.plot(train_accs)
plt.plot(test_accs)
