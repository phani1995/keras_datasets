![jpg](/assets/images/Fashion-MNIST_database_of_fashion_articles_keras_dataset_files/title_image.jpg)


In this post,  
We would like to analyse dataset **Fashion - MNIST** load it from the keras framework inbuilt function and build a neural network for it.

This Fashion-MNIST dataset was created by **Han Xiao and Kashif Rasul and Roland Vollgraf** . This dataset contains 10 fashion object images taken from Zalando magazine. This dataset is a classification dataset of 10 classes. This dataset is kind odf clone or brother to actual mnist dataset because it contains specification of same type i.e. the number of training and testing images and dimensions of images.For more information...  
[official Link](http://yann.lecun.com/exdb/mnist/)  

There is description for every line of code.

# Imports

`numpy`      import to manupulate arrays  
`pandas`     import to create and modify dataframes  
`matplotlib` to visulaize graphs  
`seaborn`    build on matploblib, higher level graph functions


```python
# Imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

# Loading the data
Here we are taking the inbuilt function of keras to load the data from the server  
The dataset file in present in the [Link to dataset in amazon server](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)  
The inbuilt code 
```python
def load_data():
    """Loads the Fashion-MNIST dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'fashion-mnist')
    base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(get_file(fname,
                              origin=base + fname,
                              cache_subdir=dirname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
```
which downloads the data and unpacks and gives back as tuples of test, train splits   

## Regarding Dataset
Fashion Mnist is data set of 10 fashoin articles namely  

| Label | Description   |
|:----  |:------------- | 
|0      | T-shirt/top   | 
|1      | Trouser       | 
|2      | Pullover      |
|3      | Dress         |
|4      | Coat          |
|5      | Sandal        |
|6      | Shirt         |
|7      | Sneaker       |
|8      | Bag           |
|9      | Ankle boot    |

[official Link](https://github.com/zalandoresearch/fashion-mnist)   

`Line 4` : created a dictionary with respective labels


```python
# Loading the dataset
from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
fashion_mnist_labels_dict = {0:"T-shirt/top",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankle boot"}
```

# Understand the dataset
Let see the size and shape of test and training tuples

``On Execution``  
There will be 60000 samples of 28*28 resolution of images in training set  
There will be 10000 samples of 28*28 resolution of images in testing set


```python
# Understanding the data
print("The number of training samples",len(x_train))
print("The number of testing samples",len(x_test))
print("The shape of training sample array",np.shape(x_train))
print("The shape of training labels",np.shape(y_train))
```

    The number of training samples 60000
    The number of testing samples 10000
    The shape of training sample array (60000, 28, 28)
    The shape of training labels (60000,)
    

# Visualizing the data
Let us try to visualise the few samples of data so, we could get an idea of how the data looks like  
`On Execution`  
We can see ten images of 


```python
# Visulizing the data
fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)
index = 0
for row in ax:
    for col in row:
        col.set_xlabel(str(fashion_mnist_labels_dict[y_train[index]]))
        col.imshow(x_train[index],cmap='gray')
        index+=1
plt.show()
print("first ten labels")
for i in range(0,10):
    print("Label value :",y_train[i])
    print("Object Name :",fashion_mnist_labels_dict[y_train[i]])
```


![png](/assets/images/Fashion-MNIST_database_of_fashion_articles_keras_dataset_files/Fashion-MNIST_database_of_fashion_articles_keras_dataset_7_0.png)


    first ten labels
    Label value : 9
    Object Name : Ankle boot
    Label value : 0
    Object Name : T-shirt/top
    Label value : 0
    Object Name : T-shirt/top
    Label value : 3
    Object Name : Dress
    Label value : 0
    Object Name : T-shirt/top
    Label value : 2
    Object Name : Pullover
    Label value : 7
    Object Name : Sneaker
    Label value : 2
    Object Name : Pullover
    Label value : 5
    Object Name : Sandal
    Label value : 5
    Object Name : Sandal
    

# Preprocessing the data

## Vairables
`image_width`,`image_height`,`image_channels` would describe the dimentions of the image  
`classes` to detemine how many catogories of samples present in out dataset. By nature mnist have 0-9 images to ten classes  

## Creating sparse vector representation
`to_categorical` is converting into one hot encoding. Means each vector is represented by one hot encoding.
0 --> [1,0,0,0,0,0,0,0,0,0]   
1 --> [0,1,0,0,0,0,0,0,0,0]  
2 --> [0,0,1,0,0,0,0,0,0,0]   
and similarly goes on 

`np.expand_dims` would just increst one dimentions in the end. Like .... [1,2,3] to [[1],[2],[3]]  

## Normalization
`Line16`,`Line17`is normalization we are divinding all the pixal values by 255. so all the numerical values are converted between 0 and 1
>Note: since its a simple dataset there is not much of processing required to attain good accuracies. For all real time datasest preprocessing like normalizing , standadising , on hot encoding, filling the missing values, transforming features, feeding the data in batches and all other type of preprocessing is required



```python
# Preprocessing the data
    
# Variables 
image_width  = 28
image_height = 28
image_channels = 1
image_shape = (image_width,image_height,image_channels)

x_train = np.expand_dims(x_train,axis=3)
x_test = np.expand_dims(x_test,axis=3)

## Creating sparse vector representation
from keras.utils import to_categorical
y_train_sparse = to_categorical(y_train)
y_test_sparse  = to_categorical(y_test)

## Normalization
x_train = x_train /255
x_test = x_test/255
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-0f675961ee66> in <module>
          7 image_shape = (image_width,image_height,image_channels)
          8 
    ----> 9 x_train = np.expand_dims(x_train,axis=3)
         10 x_test = np.expand_dims(x_test,axis=3)
         11 
    

    NameError: name 'np' is not defined


# Training varibles
These Training varbles are hyper parameters for neural network training.   
`epochs` : each epoch is forward propagation + backward propagation over the whole dataset once is called one epoch.  
`learning_rate` : the magnitude in which the weights are modified one the acquired loss.   
`learning_rate_decay` : there can be high leanring rate at the beining of the training when the loss is high. Over a period of time the learning rate can reduce for fine training of network.  
`batch_size` : the data is fed to the network in batches of 32 samples at each time. This batch feeding is done all over the whole dataset.  


```python
# Training varibles
learning_rate = 0.001
learning_rate_decay = 0.00001
batch_size = 32
epochs = 20
classes = 10
```

# Neural Netowork Model
`Line 6` : we are building a keras sequential model  
`Line 32` : we are using stochastic gradient decent optimizer  
`Line 36` : compiling the model to check if the model is build properly.  

The loss function being used is `categorical_crossentropy` since its a multi class classification



```python
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
```

# Training 
Training is the process of feeding the data to neural network and modifiying the weights of the model using the the backpropagation algorithm. The backpropagation using loss the function acquires the loss over batch size of data and does a backpropagation to modify the weights in such a way the in the next epoch the loss would be less when compared to the current epoch


```python
# Training the mode
model_history = model.fit(x=x_train, y=y_train_sparse, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test_sparse), shuffle=True)
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 15s 246us/step - loss: 2.0516 - acc: 0.2741 - val_loss: 1.4144 - val_acc: 0.5728
    Epoch 2/20
    60000/60000 [==============================] - 12s 199us/step - loss: 1.2472 - acc: 0.5253 - val_loss: 0.9791 - val_acc: 0.6371
    Epoch 3/20
    60000/60000 [==============================] - 12s 201us/step - loss: 1.0451 - acc: 0.6025 - val_loss: 0.8966 - val_acc: 0.6655
    Epoch 4/20
    60000/60000 [==============================] - 12s 200us/step - loss: 0.9589 - acc: 0.6394 - val_loss: 0.8358 - val_acc: 0.6880
    Epoch 5/20
    60000/60000 [==============================] - 12s 198us/step - loss: 0.8906 - acc: 0.6677 - val_loss: 0.7934 - val_acc: 0.7079
    Epoch 6/20
    60000/60000 [==============================] - 12s 198us/step - loss: 0.8429 - acc: 0.6838 - val_loss: 0.7588 - val_acc: 0.7193
    Epoch 7/20
    60000/60000 [==============================] - 12s 196us/step - loss: 0.8070 - acc: 0.6976 - val_loss: 0.7402 - val_acc: 0.7262
    Epoch 8/20
    60000/60000 [==============================] - 12s 196us/step - loss: 0.7764 - acc: 0.7086 - val_loss: 0.7124 - val_acc: 0.7347
    Epoch 9/20
    60000/60000 [==============================] - 12s 196us/step - loss: 0.7537 - acc: 0.7179 - val_loss: 0.6880 - val_acc: 0.7424
    Epoch 10/20
    60000/60000 [==============================] - 12s 196us/step - loss: 0.7303 - acc: 0.7265 - val_loss: 0.6741 - val_acc: 0.7483
    Epoch 11/20
    60000/60000 [==============================] - 12s 196us/step - loss: 0.7119 - acc: 0.7324 - val_loss: 0.6585 - val_acc: 0.7530
    Epoch 12/20
    60000/60000 [==============================] - 12s 200us/step - loss: 0.6986 - acc: 0.7378 - val_loss: 0.6532 - val_acc: 0.7567
    Epoch 13/20
    60000/60000 [==============================] - 12s 199us/step - loss: 0.6832 - acc: 0.7426 - val_loss: 0.6333 - val_acc: 0.7613
    Epoch 14/20
    60000/60000 [==============================] - 12s 200us/step - loss: 0.6750 - acc: 0.7458 - val_loss: 0.6244 - val_acc: 0.7686
    Epoch 15/20
    60000/60000 [==============================] - 12s 199us/step - loss: 0.6608 - acc: 0.7501 - val_loss: 0.6131 - val_acc: 0.7704
    Epoch 16/20
    60000/60000 [==============================] - 12s 199us/step - loss: 0.6514 - acc: 0.7550 - val_loss: 0.6059 - val_acc: 0.7715
    Epoch 17/20
    60000/60000 [==============================] - 12s 199us/step - loss: 0.6428 - acc: 0.7585 - val_loss: 0.5968 - val_acc: 0.7789
    Epoch 18/20
    60000/60000 [==============================] - 12s 199us/step - loss: 0.6362 - acc: 0.7613 - val_loss: 0.5909 - val_acc: 0.7832
    Epoch 19/20
    60000/60000 [==============================] - 12s 200us/step - loss: 0.6255 - acc: 0.7655 - val_loss: 0.5804 - val_acc: 0.7848
    Epoch 20/20
    60000/60000 [==============================] - 12s 200us/step - loss: 0.6213 - acc: 0.7654 - val_loss: 0.5760 - val_acc: 0.7872
    

# Results
using the trained model we try to predict what are the values of images in the test set


```python
# Results
y_pred = model.predict(x_test, batch_size = batch_size, verbose=1)
```

    10000/10000 [==============================] - 1s 72us/step
    

# Verifying the results
cheking the results how good they are with the first 10 samples.   
Plotting the graphs of test and train set accuracies and loss values. 
> NOTE: This plot is a very curicial step. These plots would tell us how good the model converges and if there is any overfitting


```python
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

```

    Ground truths of first 10 images in test set [9 2 1 1 6 1 4 6 5 7]
    Predicted values of first 10 image in test set [9 2 1 1 2 1 6 6 5 7]
    


![png](/assets/images/Fashion-MNIST_database_of_fashion_articles_keras_dataset_files/Fashion-MNIST_database_of_fashion_articles_keras_dataset_19_1.png)



![png](/assets/images/Fashion-MNIST_database_of_fashion_articles_keras_dataset_files/Fashion-MNIST_database_of_fashion_articles_keras_dataset_19_2.png)


# Visulizing the results
checking the results by visulizing them and creating a confusion matrix. The values of precession and accuracy can be obtained by the help of confusion matrix and f1 scores to compare this architecure with other architectures of neural networks


```python
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
```

                 T-shirt/top  Trouser  Pullover  Dress  Coat  Sandal  Shirt  \
    T-shirt/top          796        6        44     72    11       7     53   
    Trouser                2      941         2     38     8       1      6   
    Pullover              10        3       691     10   206       2     70   
    Dress                 32       11        35    838    52       0     29   
    Coat                   1        1       160     40   731       1     59   
    Sandal                 1        0         0      2     0     913      0   
    Shirt                226        2       302     50   184       3    215   
    Sneaker                0        0         0      0     0      38      0   
    Bag                    5        3        26      7    14       7     20   
    Ankle boot             0        1         0      0     2      13      0   
    
                 Sneaker  Bag  Ankle boot  
    T-shirt/top        0   11           0  
    Trouser            0    2           0  
    Pullover           0    8           0  
    Dress              0    3           0  
    Coat               0    7           0  
    Sandal            57    3          24  
    Shirt              0   18           0  
    Sneaker          905    1          56  
    Bag                4  912           2  
    Ankle boot        53    1         930  
    




    Text(159.0, 0.5, 'Actaul')




![png](/assets/images/Fashion-MNIST_database_of_fashion_articles_keras_dataset_files/Fashion-MNIST_database_of_fashion_articles_keras_dataset_21_2.png)

