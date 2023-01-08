
#variational autoencoder in pytorch

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from keras.datasets import mnist

# load MNIST
(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

# standardize data between 0 and 1
Xtrain = Xtrain / 255
Xtest = Xtest / 255

# resize to N x D 
Xtrain = np.resize(Xtrain, (60000,784))
Xtest = np.resize(Xtest, (10000,784))

#convert data to binary
Xtrain = (Xtrain > 0.5).astype(np.float64)
Xtest = (Xtest > 0.5).astype(np.float64)

N, D = Xtrain.shape
hidden_dim = 2


#Use GPU if possible, if not, then default to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

Xtrain = torch.from_numpy(Xtrain).float()

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.enc_d1 = nn.Sequential(nn.Linear(D, 200), nn.ReLU())
        self.enc_d2 = nn.Sequential(nn.Linear(200, 100), nn.ReLU())
        self.enc_mean = nn.Sequential(nn.Linear(100, hidden_dim))
        self.enc_var = nn.Sequential(nn.Linear(100, hidden_dim), nn.Softplus())
        
        self.dec_d1 = nn.Sequential(nn.Linear(hidden_dim, 100), nn.ReLU())
        self.dec_d2 = nn.Sequential(nn.Linear(100, 200), nn.ReLU())
        self.dec_d3 = nn.Sequential(nn.Linear(200, D), nn.Sigmoid())
        
    def forward(self, x):
        out = self.enc_d1(x) #N x 200
        out = self.enc_d2(out) #N x 100
        mean = self.enc_mean(out) #N x hidden_dim
        var = self.enc_var(out) #N x hidden_dim
        
        #reparameterization trick
        samples = torch.normal(0, 1, size=(mean.shape[0], mean.shape[1])).to(device) #N x hidden_dim
        transformed = var * samples + mean #N x hidden_dim
        
        out = self.dec_d1(transformed) #N x 100
        out = self.dec_d2(out) #N x 200
        out = self.dec_d3(out) #N x 784
        
        return mean, var, transformed, out
    
    #get prior predictive prob
    def prior_forward(self,latent):
        out = self.dec_d1(latent) #N x 100
        out = self.dec_d2(out) #N x 200
        out = self.dec_d3(out) #N x 784
        return out
        
model = VAE()
model.to(device)

from torchinfo import summary
summary(model, input_size=(64, 784))

#
def elbo(targ, pred, mean, var):
    bce = nn.BCELoss(reduce = False)
    bceloss = bce(pred, targ)
    #sum along all dimensions for each sample, and get mean for all samples
    reconstruction_loss = torch.mean(torch.sum(bceloss, axis = 1) ) 
    
    
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    kl_loss = -0.5 * (1 + torch.log(var) - (torch.square(mean) + torch.square(var)))
    kl_loss = torch.mean(torch.sum(kl_loss, axis=1))
    
    total_loss = reconstruction_loss + kl_loss
    return total_loss
    
optimizer = optim.Adam(model.parameters())


#train, and get train cost
def train(model, optimizer, inputs):
    model.train()
    
    inputs = Variable(inputs, requires_grad = False)
    inputs.to(device)

    optimizer.zero_grad()
    
    mean, var, transformed, pred = model.forward(inputs)
    
    #loss defined in here
    out = elbo(inputs, pred, mean, var).to(device)
    
    out.backward() 
    optimizer.step()  
    return out.item()

#params
B = 200
N = Xtrain.shape[0]
n_batches = int(N//B)
n_iter = 40

#store costs and acc
train_costs = []

#train
for i in range(n_iter):
    train_cost = 0.
    
    for j in range(n_batches):
        Xbatch = Xtrain[j*B:(j+1)*B].to(device)
        
        train_cost += train(model, optimizer, Xbatch)
        
    print('iter: ', i,
          'train cost: ', train_cost / n_batches)    
    
    train_costs.append(train_cost / n_batches)





#get posterior predictive sample p(x_new|x_input)
idx = np.random.randint(len(Xtrain))
sample = Xtrain[idx:idx+1,:]

mean, var, transformed, posterior_pred = model.forward(sample.to(device))
posterior_pred = posterior_pred.cpu().detach().numpy().reshape((28,28))

posterior_pred_sample = np.random.binomial(n = 1, p = posterior_pred, size = posterior_pred.shape)

plt.subplot(131)
plt.imshow(sample.reshape((28,28)))
plt.gray()

plt.subplot(132)
plt.imshow(posterior_pred)
plt.gray()

plt.subplot(133)
plt.imshow(posterior_pred_sample)
plt.gray()





#get prior predictive sample p(x_new|z)
latent_x = np.linspace(-3,3,30)
latent_y = np.linspace(-3,3,30)

all_images = np.zeros((28 * 30, 28 * 30))

for i in range(len(latent_y)):
    for j in range(len(latent_x)):
        meanx = latent_x[j]
        meany  = latent_y[i]
        latent = np.array([meanx,meany])
        latent = torch.from_numpy(latent).float()

        prior_pred = model.prior_forward(latent.to(device))
        prior_pred = prior_pred.cpu().detach().numpy().reshape((28,28))

        prior_pred_sample = np.random.binomial(n = 1, p = prior_pred, size = prior_pred.shape)
        
        all_images[i*28:(i+1)*28, j*28:(j+1)*28] = prior_pred_sample

plt.imshow(all_images)
plt.gray()



#view latent space

mean, var, transformed, pred = model.forward(Xtrain.to(device))
latent = mean.cpu().detach().numpy()

plt.scatter(latent[:,0], latent[:,1], c = Ytrain)
plt.jet()
