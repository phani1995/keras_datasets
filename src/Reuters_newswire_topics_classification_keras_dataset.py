# -*- coding: utf-8 -*-

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
from keras.datasets import reuters
(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",num_words=10000,skip_top=0,maxlen=None,test_split=0.2,seed=113,start_char=1,oov_char=2,index_from=3)

# Understanding the data
print("The number of training samples",len(x_train))
print("The number of testing samples",len(x_test))
print("The shape of training samples array",np.shape(x_train))
print("The shape of training samples labels", np.shape(y_train))
print("First element in x_train ,its type :",type(x_train[0]),"it's shape :",np.shape(x_train[0]))

# Data Visulization

# A dictionary mapping words to an integer index
word_index = reuters.get_word_index()
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

##Varibles
classes = 46

## Padding
from keras.preprocessing.sequence import pad_sequences
x_train_padded = pad_sequences(x_train,value=0,padding='post',maxlen=256)
x_test_padded =  pad_sequences(x_test,value=0,padding='post',maxlen=256)

## Creating sparse vector representation
from keras.utils import to_categorical
y_train_sparse = to_categorical(y_train,num_classes=classes)
y_test_sparse = to_categorical(y_test,num_classes=classes)


# Training varibles
learning_rate = 0.0005
learning_rate_decay = 0.00001
batch_size = 512
epochs = 30


# Building the model
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

from keras.models import Sequential
from keras.layers import Embedding,GlobalAveragePooling1D,Dense,Dropout
from keras.optimizers import Adam

model = Sequential()
model.add(Embedding(vocab_size, 512))
model.add(GlobalAveragePooling1D())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(46, activation = 'sigmoid'))

model.summary()

# optimizer
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learning_rate_decay, amsgrad=False)

# Model Compilation
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

# Training the model
model_history = model.fit(x_train_padded,y_train_sparse,epochs=epochs,batch_size=batch_size,validation_data=(x_test_padded, y_test_sparse),verbose=1)

# Results
y_pred = model.predict(x_test_padded)

# Verifying the results
print("Ground truths of first 10 images in test set",np.array(y_test[0:10]))
print("Predicted values of first 10 image in test set",np.argmax(y_pred,axis=1))

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
y_pred = np.argmax(y_pred,axis=1)
y_pred = pd.Series(y_pred, name='Predicted')
y_test = pd.Series(y_test, name='Actual')
df_confusion  = pd.crosstab(y_test,y_pred, rownames=['Actual'], colnames=['Predicted'])
print(df_confusion)
plt.figure(figsize = (20,20))
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=18)
plt.title('Confusion Matrix',fontsize=20)
sns.heatmap(df_confusion, annot=True,fmt="d")
