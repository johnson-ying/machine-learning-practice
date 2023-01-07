
#variational autoencoder in theano
#note: theano is not optimized for training speed, just practiced structure

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from keras.datasets import mnist
from theano.tensor.shared_randomstreams import RandomStreams

# load MNIST
(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

# standardize data between 0 and 1
Xtrain = Xtrain / 255
Xtest = Xtest / 255

# resize to N x D 
Xtrain = np.resize(Xtrain, (60000,784))
Xtest = np.resize(Xtest, (10000,784))

class Layer():
    def __init__(self, M1, M2, activation):
        self.w = theano.shared(np.random.randn(M1,M2)/np.sqrt(M1))
        self.b = theano.shared(np.zeros((M2)))
        self.params = [self.w, self.b]
        self.activation = activation
    
    def forward(self, X):
        if self.activation == None:
            return X.dot(self.w) + self.b
        else:
            return self.activation(X.dot(self.w) + self.b)
    
class VAE():
    def __init__(self, encoder_sizes, decoder_sizes):
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        
    def fit(self, Xtrain, Ytrain, lr = 1e-4, batch_sz = 200, n_iter = 10):
        K = Xtrain.shape[1]
        
        n_batch = Xtrain.shape[0]//batch_sz
        
        #create encoder and decoder layers, and store their params
        self.encoder = []
        self.decoder = []
        self.params = []
        
        #encoder
        M1 = K
        for n in range(len(self.encoder_sizes)):   
            M2 = self.encoder_sizes[n]
            
            #if it is the final output layer
            if n == len(self.encoder_sizes) - 1:
                #layer output should be twice the dimension, since we'll output mean and variance
                layer = Layer(M1, M2 * 2, None) #we'll manually apply the activation functions
                self.encoder.append(layer)
                self.params += layer.params
                M1 = M2
                half = M2 #use for later
                
            #if the layers preceding it
            else:
                layer = Layer(M1, M2, T.nnet.relu)
                self.encoder.append(layer)
                self.params += layer.params
                M1 = M2
        
        #decoder
        for M2 in self.decoder_sizes:
            layer = Layer(M1, M2, T.nnet.relu)
            self.decoder.append(layer)
            self.params += layer.params
            M1 = M2
            
        #final layer
        final_layer = Layer(M1, K, T.nnet.sigmoid)
        self.params += final_layer.params
        
        #
        #theano variables
        thX = T.matrix('inputs') #also the targets 
        z_sample = T.matrix('sampled z')
        
        #encoder pass
        z = thX
        for layer in self.encoder:
            z = layer.forward(z)
        
        #dont use activation function to get mean, but use softplus for var
        mean = z[:, :half] #N x _
        #i called it var, but really, this is the standard deviation
        var = T.nnet.softplus(z[:, half:]) + 1e-8 #N x _
        
        #use built in theano random number sampler to sample from a gaussian N(0,1)
        self.rng = RandomStreams()
        eps = self.rng.normal((mean.shape[0], half))
        transformed = mean + var * eps
        
        #decoder pass
        z2 = transformed    
        for layer in self.decoder:
            z2 = layer.forward(z2)
        
        out = final_layer.forward(z2)
        
        # KL divergence
        # use this --> https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        #assume each dimension D is a separate univariate gaussian z 
        #we want the KL divergence between z and the standard normal (0,1)
        kl = -T.log(var) + 0.5*(var**2 + mean**2) - 0.5
        kl = T.sum(kl, axis=1) #and we sum along each dimension

        #expected log likelihood is binary cross entropy
        expected_log_likelihood = T.sum((thX * out + (1-thX)*T.log(1-out)), axis = 1)
        
        #cost function is the difference between expected log likelihood and KL
        #and sum this for all samples
        elbo = T.sum(expected_log_likelihood - kl)
        
        #elbo is gradient ascent, so add a negative to do gradient descent
        elbo = -elbo
        
        #gradients
        grads = T.grad(elbo, self.params)

        #updates
        # regular gradient descent seems to lead to unstable training
        # updates = [(p, p - lr*g) for p, g in zip(self.params, grads)]
        
        #implement adam or anything that has adaptive learning rates
        beta1 = 0.9
        beta2 = 0.99
        first_moments = [theano.shared(np.ones_like(p.get_value())) for p in self.params]
        second_moments = [theano.shared(np.ones_like(p.get_value())) for p in self.params]
        
        new_first_moments = [beta1 * moment1 + (1-beta1) * g for moment1, g in zip(first_moments, grads)]
        new_second_moments = [beta2 * moment2 + (1-beta2) * g * g for moment2, g in zip(second_moments, grads)]
        
        #bias correction
        first_moments_cor = [moment1 / (1 - beta1 ** 2) for moment1 in new_first_moments]
        second_moments_cor = [moment2 / (1 - beta2 ** 2) for moment2 in new_second_moments]

        updates = [
            (m1, new_m1) for m1, new_m1 in zip(first_moments, new_first_moments)
        ] + [
            (m2, new_m2) for m2, new_m2 in zip(second_moments, new_second_moments)
        ] + [
            (p, p - lr * m/T.sqrt(v + 1e-10)) for p, m, v in zip(self.params, first_moments_cor, second_moments_cor)
        ]

        #train op
        train_op = theano.function(inputs = [thX],
                                   updates = updates,
                                   outputs = elbo)
        
        #get the prior probability for a given z
        zz = z_sample #1 x _
        for layer in self.decoder:
            zz = layer.forward(zz)
        sampled = final_layer.forward(zz) #N x _
        
        self.get_x_from_z = theano.function(inputs = [z_sample],
                                       outputs = sampled)
        
        #get posterior probability for input x
        self.get_x_from_x = theano.function(inputs = [thX],
                                       outputs = out)
        
        #get latent representation z from an input x
        self.get_z_from_x = theano.function(inputs = [thX],
                                       outputs = mean) #get the transformed mean
        
        #train loop
        self.costs = []
        for i in range(n_iter):
            for j in range(n_batch):
                Xbatch = Xtrain[j*batch_sz: (j+1)*batch_sz, :]                
                cost = train_op(Xbatch)
                self.costs.append(cost)

                print("iter: ", i, "batch #: ", j, "cost: ", cost)
                
        plt.plot(self.costs)

#make data binary
Xtrain = (Xtrain > 0.5).astype(np.float64)
Xtest = (Xtest > 0.5).astype(np.float64)
#just do on subset of train data to make this go faster
Xtrain = Xtrain[0:10000,:]
Ytrain = Ytrain[0:10000]


latent_dim = 2

encoder_sizes = [200,100,latent_dim]
decoder_sizes = [100,200]

model = VAE(encoder_sizes, decoder_sizes)
model.fit(Xtest, Ytest, lr = 1e-2, batch_sz = 2000, n_iter = 5)

#get prior predictive probabilities and sample
prior_pred_prob = model.get_x_from_z(np.random.randn(1,latent_dim)).reshape((28,28))

rng = RandomStreams()
prior_pred_sample = rng.binomial(size=prior_pred_prob.shape, n=1, p=prior_pred_prob)

plt.subplot(121)
plt.imshow(prior_pred_sample.eval())
plt.gray()
plt.subplot(122)
plt.imshow(prior_pred_prob)
plt.gray()

#get posterior predictive probabilities and sample
idx = np.random.choice(len(Xtrain), 1)[0]
sample = Xtrain[idx,:]
posterior_pred_prob = model.get_x_from_x(sample.reshape((28,28)))

rng = RandomStreams()
posterior_pred_sample = rng.binomial(size=prior_pred_prob.shape, n=1, p=posterior_pred_prob)

plt.subplot(121)
plt.imshow(posterior_pred_sample.eval())
plt.gray()
plt.subplot(122)
plt.imshow(posterior_pred_prob)
plt.gray()

#get latent representations
model.get_z_from_x(Xtrain)
