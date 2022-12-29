
#visualize the learned features in CNN architectures
#load pre-trained weights so we dont have to train again

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

#load pre-trained weights
model.load_state_dict(torch.load('model weights'))

model.to(device)

# loss = torch.nn.CrossEntropyLoss(size_average=True)
# loss.to(device)

# optimizer = optim.Adam(model.parameters())

# #train, and get train cost
# def train(model, loss, optimizer, inputs, targets):
#     model.train()
    
#     inputs = Variable(inputs, requires_grad = False)
#     inputs.to(device)
#     targets = Variable(targets, requires_grad = False)
#     targets.to(device)

#     optimizer.zero_grad()
    
#     out = model.forward(inputs)
#     out2 = loss.forward(out, targets) 
    
#     out2.backward() 
#     optimizer.step()  
#     train_loss = out2.item()
#     del out
#     return train_loss

#convert data to tensors - do not store them in GPU, only store when training
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()

# #params
# B = 64
# N = Xtrain.shape[0]
# n_batches = int(N//B)
# n_iter = 10

# #store costs and acc
# train_costs = []

# #train
# for i in range(n_iter):
#     train_cost = 0.
#     test_cost = 0.
#     model.to(torch.device('cuda')) 
#     for j in range(n_batches):
#         Xbatch = Xtrain[j*B:(j+1)*B].to(device)
#         Ybatch = Ytrain[j*B:(j+1)*B].to(device)
        
#         train_cost += train(model, loss, optimizer, Xbatch, Ybatch)
    
#     print('iter: ', i, 'train cost: ', train_cost / n_batches)        
#     train_costs.append(train_cost / n_batches)

# torch.save(model.state_dict(), 'model weights')



from torchvision.models.feature_extraction import create_feature_extractor

return_nodes = {
    "conv1": "f1",
    "conv2": "f2",
    "conv3": "f3",
    "conv4": "f4",
    "conv5": "f5",
    "conv6": "f6",
}
model2 = create_feature_extractor(model, return_nodes=return_nodes)






#get random image
randomid = np.random.randint(0, len(Xtest))
img = np.reshape(Xtest[randomid], (1,1,28,28))
plt.imshow(img[0,0,:,:])
plt.gray()

#get features 
img = torch.from_numpy(img).float().to(device)
features = model2.forward(img)

f1 = features['f1'].cpu().detach().numpy()
f2 = features['f2'].cpu().detach().numpy()
f3 = features['f3'].cpu().detach().numpy()
f4 = features['f4'].cpu().detach().numpy()
f5 = features['f5'].cpu().detach().numpy()
f6 = features['f6'].cpu().detach().numpy()

#first conv
fig1 = plt.figure("Conv 1")
for i in range(f1.shape[-1]):
    plt.subplot(8,8,i+1)
    plt.imshow(f1[0,i,:,:])
    plt.axis('off') 

#second conv
fig2 = plt.figure("Conv 2")
for i in range(f2.shape[-1]):
    plt.subplot(8,8,i+1)
    plt.imshow(f2[0,i,:,:])
    plt.axis('off') 

#third conv
fig3 = plt.figure("Conv 3")
for i in range(f3.shape[-1]):
    plt.subplot(12,12,i+1)
    plt.imshow(f3[0,i,:,:])
    plt.axis('off') 

#fouth conv
fig4 = plt.figure("Conv 4")
for i in range(f4.shape[-1]):
    plt.subplot(12,12,i+1)
    plt.imshow(f4[0,i,:,:])
    plt.axis('off') 
