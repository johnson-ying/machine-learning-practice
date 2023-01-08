
#variational autoencoder in keras
#using implementation from:
#https://keras.io/examples/generative/vae/

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import tensorflow as tf

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

#build encoder
i = Input(shape = (D,))
x = Dense(200, activation = 'relu')(i)
x = Dense(100, activation = 'relu')(x)
mean_hidden = Dense(hidden_dim)(x)
std_hidden = Dense(hidden_dim, activation = 'softplus')(x)

#sample
class GaussianSample(keras.layers.Layer):
    #reparameterization trick
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        # return z_mean + keras.exp(0.5 * z_log_var) * epsilon
        return z_mean + z_log_var * epsilon
    
z = GaussianSample()([mean_hidden, std_hidden])
encoder = Model(inputs = i, outputs = [mean_hidden, std_hidden, z])

encoder.summary()


#build decoder
latent_i = Input(shape = (hidden_dim))
x = Dense(100, activation = 'relu')(latent_i)
x = Dense(200, activation = 'relu')(x)
out = Dense(D, activation = 'sigmoid')(x)
decoder = Model(inputs = latent_i, outputs = out)

decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # reconstruction_loss = keras.backend.sum((data * reconstruction + (1-data)*keras.backend.log(1-reconstruction)), axis = 1)
            
            # kl_loss = -keras.backend.log(z_log_var) + 0.5*(z_log_var**2 + z_log_var**2) - 0.5
            # kl_loss = keras.backend.sum(kl_loss, axis=1) #and we sum along each dimension
            # total_loss = -keras.backend.sum(reconstruction_loss - kl_loss)
            
            reconstruction_loss = keras.backend.mean(
                keras.backend.sum(
                    keras.losses.binary_crossentropy(data, reconstruction)
                )
            )
            
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_loss = -0.5 * (1 + keras.backend.log(z_var) - (keras.backend.square(z_mean) + keras.backend.square(z_var)))
            kl_loss = keras.backend.mean(keras.backend.sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(Xtrain, epochs=30, batch_size=128)






#get posterior predictive sample p(x_new|x_input)
idx = np.random.randint(len(Xtrain))
sample = Xtrain[idx:idx+1,:]

latent = encoder.predict(sample)
posterior_pred = decoder(latent[2])
posterior_pred = posterior_pred.numpy().reshape((28,28))

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
latent = np.random.randn(1,hidden_dim)
latent = np.random.randn(1,hidden_dim) * np.array([.2, .001]) + np.array([-2, 1])

prior_pred = decoder(latent)
prior_pred = prior_pred.numpy().reshape((28,28))

prior_pred_sample = np.random.binomial(n = 1, p = prior_pred, size = prior_pred.shape)

plt.subplot(121)
plt.imshow(prior_pred)
plt.gray()

plt.subplot(122)
plt.imshow(prior_pred_sample)
plt.gray()



#view latent space

latent = encoder.predict(Xtrain)
latent = latent[0]

plt.scatter(latent[:,0], latent[:,1], c = Ytrain)
plt.jet()
