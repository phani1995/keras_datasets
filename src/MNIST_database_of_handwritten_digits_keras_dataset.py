
# Imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# Understanding the data
print("The size of training samples is ",len(x_train))
print("The size of testing samples is ",len(x_test))
print("The shape of training samples array is",np.shape(x_train))
print("The shape of training labels is ",np.shape(y_train))

# Visualizing the data
fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)
index = 0
for row in ax:
    for col in row:
        col.imshow(x_train[index],cmap='gray')
        index+=1
#plt.show()
print("first ten labels",y_train[0:10])

# Preprocessing the data

# Vairables
image_width = 28
image_height = 28
image_channels = 1
input_shape = (image_width,image_height,image_channels)
classes =  10
# Creating sparse vector representation
from keras.utils import to_categorical
y_train_sparse = to_categorical(y_train,num_classes=classes)
y_test_sparse = to_categorical(y_test,num_classes=classes)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

x_train = x_train/255 # Normalizing the x_trian
x_test = x_test/255 # Normalizing the X_test



# Training varibles
epochs = 10
learning_rate = 0.001
learning_rate_decay = 0.000001
batch_size = 32

# Building the model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Activation,Dense
from keras.optimizers import SGD

model = Sequential()

# Layer 1
model.add(Conv2D(64,kernel_size=(3,3), strides=(1, 1), padding='valid', data_format="channels_last", input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# Layer 2
model.add(Conv2D(32,kernel_size =(3,3), strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))

# Layer 3
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))

# Layer 4
model.add(Dense(256))
model.add(Activation('relu'))

# Layer 5
model.add(Dense(10,activation='softmax'))

# optimizer
optimizer = SGD(lr = learning_rate, momentum=0.0, decay=learning_rate_decay, nesterov=True)

# Model Compilation
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model_history = model.fit(x=x_train, y=y_train_sparse, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test_sparse), shuffle=True, initial_epoch=1)

# Results
y_pred = model.predict(x_test,  verbose=1)

# Verifying the results
print("Ground truths of first 10 images in test set",np.array(y_test[0:10]))
print("Predicted values of first 10 image in test set",np.argmax(y_pred[0:10],axis=1))

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

