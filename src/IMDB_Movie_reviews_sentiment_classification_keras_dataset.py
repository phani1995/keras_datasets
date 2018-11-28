# -*- coding: utf-8 -*-

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000,skip_top=0,maxlen=None,seed=113,start_char=1,oov_char=2,index_from=3)

#Understanding the data
print("The number of training samples",len(x_train))
print("The number of testing samples",len(x_test))
print("The shape of training samples array",np.shape(x_train))
print("The shape of training samples labels", np.shape(y_train))
print("First element in x_train ,its type :",type(x_train[0]),"it's shape :",np.shape(x_train[0]))

# Data Visulization

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # Reversed dictionary

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

## Viewing the review's
print('\nFirst Review \n')
print(decode_review(x_train[0]))
print('\nIts label :',y_train[0])

print('\nSecond Review \n')
print(decode_review(x_train[1]))
print('\nIts label :',y_train[1])

# Data Preprocessing
## Padding
from keras.preprocessing.sequence import pad_sequences
x_train_padded = pad_sequences(x_train,value=0,padding='post',maxlen=256)
x_test_padded =  pad_sequences(x_test,value=0,padding='post',maxlen=256)

# Training varibles
learning_rate = 0.00001
learning_rate_decay = 0.00001
batch_size = 512
epochs = 60
classes = 2

# Building the model
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

from keras.models import Sequential
from keras.layers import Embedding,GlobalAveragePooling1D,Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Embedding(vocab_size, 16))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# optimizer
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learning_rate_decay, amsgrad=False)

# Model Compilation
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

# Training the model
model_history = model.fit(x_train_padded,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test_padded, y_test),verbose=1)

# Results
y_pred = model.predict(x_test_padded)

def thershold_function(x): # Function to convert sigmoid outputs to 0's and 1's
    if x>0.5:
        return 1
    else :
        return 0
thershold = np.vectorize(thershold_function)

# Results
y_pred = thershold(y_pred)

# Verifying the results
print("Ground truths of first 10 images in test set",np.array(y_test[0:10]))
print("Predicted values of first 10 image in test set",y_pred[0:10].ravel())

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.plot(loss,label='train')
plt.plot(val_loss,label='test')
plt.title('loss Graph')
plt.ylabel('precentage')
plt.xlabel('epochs')
plt.legend()
plt.show()

acc = model_history.history['acc']
val_acc = model_history.history['val_acc']
plt.plot(acc,label='train')
plt.plot(val_acc,label='test')
plt.title('Accuracy Graph')
plt.ylabel('precentage')
plt.xlabel('epochs')
plt.legend()
plt.show()

# Visulizing the results
y_pred = y_pred.ravel()
y_pred = pd.Series(y_pred, name='Predicted')
y_test = pd.Series(y_test, name='Actual')
df_confusion  = pd.crosstab(y_test,y_pred, rownames=['Actual'], colnames=['Predicted'])
print(df_confusion)
plt.figure(figsize = (20,20))
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
plt.title('Confusion Matrix',fontsize=20)
sns.heatmap(df_confusion, annot=True,fmt="d")

