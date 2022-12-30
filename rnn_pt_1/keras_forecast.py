
#try 1-step or multi-step forecasting on time series data
#data from https://www.kaggle.com/datasets/shenba/time-series-datasets

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, SimpleRNN, GRU, LSTM
import pandas as pd

#load text, get targets
data = pd.read_csv('Electric_Production.csv', encoding = "ISO-8859-1") 
data = data['IPG2211A2N'].to_numpy()
data = (data - np.mean(data)) / np.std(data) #standardize

#visualize
plt.plot(data)

#predict next value using previous 5 values
X = []
Y = []
T = 5  
D = 1

for i in range(0,len(data)-T):
    X.append(data[i:i+T])
    Y.append(data[i+T]) 

#reshape X data into N x T x D
X = np.array(X).reshape((-1,T,D))
Y = np.array(Y)

N = X.shape[0]
half = N//2

#first half is train data, second half is test data
Xtrain, Ytrain = X[:half], Y[:half]
Xtest, Ytest = X[half:], Y[half:]

#model
# i = Input(shape = (T,D))
# x = SimpleRNN(20, activation = 'relu')(i)
# x = Dense(1)(x)

i = Input(shape = (T,D))
x = LSTM(75, activation = 'relu')(i)
x = Dense(1)(x)

model = Model(inputs = i, outputs = x)

model.compile(
  loss='MSE',
  optimizer='Adam',
  metrics=['accuracy']
)

r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=55, batch_size=64)

# loss
plt.plot(r.history['loss'], label='train')
plt.plot(r.history['val_loss'], label='test')
plt.legend()


#1 step forecast
#results look decent but misleading, and not really helpful since this is just a 1-step forecast
predictions = []
for i in range(len(Ytest)):
    #get prediction
    p = model.predict(Xtest[i].reshape(-1,T,1))[0][0]
    predictions.append(p)

plt.plot(Ytest, label = 'validation')
plt.plot(predictions, label = 'prediction')



#Multi step forecast
#using predicted values to predict next ones, which is technically more useful
#results don't look good 
last_x = Xtest[1].reshape(-1,T,1)
predictions = []
while len(predictions) < len(Ytest):
    #get prediction
    p = model.predict(last_x)[0][0]
    predictions.append(p)
    
    #update last_x to incorporate this newly predicted v alue
    last_x = np.roll(last_x,-1)
    last_x[-1] = p

plt.plot(Ytest, label = 'validation')
plt.plot(predictions, label = 'prediction')
