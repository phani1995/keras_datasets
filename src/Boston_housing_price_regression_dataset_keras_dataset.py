# -*- coding: utf-8 -*-

# Imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
     
# Understanding the data
print("The number of training samples",len(x_train))
print("The number of testing samples",len(x_test))
print("The shape of training samples array",np.shape(x_train))
print("The shape of training samples labels", np.shape(y_train))

# Visualizing the data

df = pd.DataFrame(x_train, columns=columns[:-1])
df.head()

## Labels 
print("{} values :".format(columns[-1]),y_train[0:10]) 


# Preprocessing the data

## Variables
image_width = 32
image_height = 32
image_channels = 3 
image_shape = (image_width,image_height,image_channels)

## Normalization
mean = x_train.mean(axis=0)        #Train Data
std = x_train.std(axis=0)
train_data = (x_train - mean) / std

mean = y_train.mean(axis=0)        #Train Labels Data
std = y_train.std(axis=0)
y_train = (y_train - mean) / std

mean = x_test.mean(axis=0)        #Test Data
std = x_test.std(axis=0)
train_data = (x_test - mean) / std

mean = y_test.mean(axis=0)        #Test Labels Data
std = y_test.std(axis=0)
y_test = (y_test - mean) / std

# Training varibles
learning_rate = 0.0001
learning_rate_decay = 0.000001
batch_size = 32
epochs = 20
classes = 10

# Building the model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(64, activation = 'relu',input_shape=(x_train.shape[1],)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1))

# Optimizer
optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=learning_rate_decay)

# Compiling the model
model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])

# Training
model_history = model.fit(x_train, y_train, epochs=epochs,validation_split=0.2, verbose=1)

# Results
y_pred = model.predict(x=x_test, batch_size=batch_size, verbose=1)

# Verifying the results
print("Ground truths of first 10 images in test set",np.array(y_test[0:10]))
print("Predicted values of first 10 image in test set",np.argmax(y_pred[0:10],axis=1))

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.plot(loss,label='train')
plt.plot(val_loss,label='test')
plt.title('loss Graph')
plt.ylabel('value')
plt.xlabel('epochs')
plt.legend()
plt.show()

acc = model_history.history['mean_absolute_error']
val_acc = model_history.history['val_mean_absolute_error']
plt.plot(acc,label='train')
plt.plot(val_acc,label='test')
plt.title('mean_absolute_error Graph')
plt.ylabel('value')
plt.xlabel('epochs')
plt.legend()
plt.show()
