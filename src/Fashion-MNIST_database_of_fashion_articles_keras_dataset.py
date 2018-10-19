# Imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
fashion_mnist_labels_dict = {0:"T-shirt/top",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankle boot"}

# Understanding the data
print("The number of training samples",len(x_train))
print("The number of testing samples",len(x_test))
print("The shape of training sample array",np.shape(x_train))
print("The shape of training labels",np.shape(y_train))

# Visulizing the data
fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)
index = 0
for row in ax:
    for col in row:
        col.imshow(x_train[index],cmap='gray')
        index+=1
#plt.show()
print("first ten labels")
for i in range(0,10):
    print("Label value :",y_train[i])
    print("Object Name :",fashion_mnist_labels_dict[y_train[i]])

# Variables 
image_width  = 28
image_height = 28
image_channels = 1
image_shape = (image_width,image_height,image_channels)

# Training varibles
learning_rate = 0.001
learning_rate_decay = 0.00001
batch_size = 32
epochs = 20
classes = 10

# Data preporcessing
x_train = np.expand_dims(x_train,axis=3)
x_test = np.expand_dims(x_test,axis=3)

from keras.utils import to_categorical
y_train_sparse = to_categorical(y_train, num_classes=classes)
y_test_sparse  = to_categorical(y_test, num_classes=classes)

x_train = x_train /255
x_test = x_test/255

# Building the models
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import SGD

model = Sequential()

# Layer 1
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='valid', data_format="channels_last",
                input_shape=image_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))

# Layer 2
model.add(Conv2D(filters = 64, kernel_size=(3,3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))

# Layer 3
model.add(Flatten(data_format="channels_last"))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))

# Layer 4
model.add(Dense(10,activation="softmax"))

sgd_optimizer = SGD(lr = learning_rate, decay = learning_rate_decay)

model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Training the mode
model_history = model.fit(x=x_train, y=y_train_sparse, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test_sparse), shuffle=True)

# Results
y_pred = model.predict(x_test, batch_size = batch_size, verbose=1)

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
y_pred = pd.Series(y_pred,name = "predicted")
y_test = pd.Series(y_test,name = "Actual")
df_confusion  = pd.crosstab(y_test,y_pred)
df_confusion.columns = [i for i in list(fashion_mnist_labels_dict.values())]
df_confusion.index = [i for i in list(fashion_mnist_labels_dict.values())]

print(df_confusion)
plt.figure(figsize = (20,20))
plt.title('Confusion Matrix',fontsize=20)
sns.heatmap(df_confusion, annot=True,fmt="d")
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actaul', fontsize=18)








