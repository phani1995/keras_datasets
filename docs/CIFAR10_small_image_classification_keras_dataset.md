![jpg](/assets/images/CIFAR10_small_image_classification_keras_dataset_files/title_image.jpg)


In this post,  
We would like to analyse dataset **CIFAR10 small image classification** load it from the keras framework inbuilt function and build a neural network for it.

This Cifar10 dataset was created by **Alex Krizhevsky** a researcher at google. This dataset is a classification dataset of 10 classes. For more information...  
[official Link](https://www.cs.toronto.edu/~kriz/cifar.html)  

There is description for every line of code.

# Code

# Imports

`numpy`      package for array handling  
`pandas`     import to create and modify dataframes     
`matplotlib` package for data visulization  
`seaborn`    build on matploblib, higher level graph functions


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

# Loading the data
Here we are taking the inbuilt function of keras to load the data from the server  
The dataset file in present in the [Link to dataset in amazon server](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)  
The inbuilt code 
```python
def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
```
## Regarding Dataset
cifar10(Canadian Institute For Advanced Research) is dataset of 10 objectst namely

| Label | Description   |
|:----  |:------------- | 
|0      | airplane      | 
|1      | automobile    | 
|2      | bird          |
|3      | cat           |
|4      | deer          |
|5      | dog           |
|6      | frog          |
|7      | horse         |
|8      | ship          |
|9      | truck         |

[official Link](https://www.cs.toronto.edu/~kriz/cifar.html)   
`Line 4` : created a dictionary with respective labels


```python
# Loading the dataset
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cifar10_labels_dict = {0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
```

    Using TensorFlow backend.
    

# Understand the dataset
Let see the size and shape of test and training tuples

``On Execution``  
There will be 50000 samples of 28*28 resolution of images in training set  
There will be 10000 samples of 28*28 resolution of images in testing set


```python
# Understanding the data
print("The number of training samples",len(x_train))
print("The number of testing samples",len(x_test))
print("The shape of training samples array",np.shape(x_train))
print("The shape of training samples labels", np.shape(y_train))
```

    The number of training samples 50000
    The number of testing samples 10000
    The shape of training samples array (50000, 32, 32, 3)
    The shape of training samples labels (50000, 1)
    

# Visualizing the data
Let us try to visualise the few samples of data so, we could get an idea of how the data looks like  
`On Execution`  
We can see ten images of 


```python
# Visualizing the data
fig, ax = plt.subplots(nrows=2, ncols=5)
index = 0
for row in ax:
    for col in row:
        col.set_title(cifar10_labels_dict[y_train[index][0]])
        col.imshow(x_train[index])
        index+=1
plt.show()

print("first ten labels")
for i in range(0,10):
    print("Label value :",y_train[i][0])
    print("Object Name :",cifar10_labels_dict[y_train[i][0]])
```


![png](/assets/images/CIFAR10_small_image_classification_keras_dataset_files/CIFAR10_small_image_classification_keras_dataset_7_0.png)


    first ten labels
    Label value : 6
    Object Name : frog
    Label value : 9
    Object Name : truck
    Label value : 9
    Object Name : truck
    Label value : 4
    Object Name : deer
    Label value : 1
    Object Name : automobile
    Label value : 1
    Object Name : automobile
    Label value : 2
    Object Name : bird
    Label value : 7
    Object Name : horse
    Label value : 8
    Object Name : ship
    Label value : 3
    Object Name : cat
    

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

## Variables
image_width = 32
image_height = 32
image_channels = 3 
image_shape = (image_width,image_height,image_channels)

## Normalization
x_train  = x_train /255
x_test = x_test /255

## Creating sparse vector representation
from keras.utils import to_categorical
y_train_sparse = to_categorical(y_train)
y_test_sparse = to_categorical(y_test)

```

# Training varibles
These Training varbles are hyper parameters for neural network training.   
`epochs` : each epoch is forward propagation + backward propagation over the whole dataset once is called one epoch.  
`learning_rate` : the magnitude in which the weights are modified one the acquired loss.   
`learning_rate_decay` : there can be high leanring rate at the beining of the training when the loss is high. Over a period of time the learning rate can reduce for fine training of network.  
`batch_size` : the data is fed to the network in batches of 32 samples at each time. This batch feeding is done all over the whole dataset.  


```python
# Training varibles
classes = 10
epochs = 20
learning_rate = 0.05
learning_rate_decay = 0.0001
batch_size = 32
```

# Neural Netowork Model
`Line 6` : we are building a keras sequential model  
`Line 32` : we are using stochastic gradient decent optimizer  
`Line 36` : compiling the model to check if the model is build properly.  

The loss function being used is `categorical_crossentropy` since its a multi class classification



```python
# Buliding the model 
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.optimizers import SGD

model = Sequential()

# Layer 1
model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = (1, 1), padding = 'valid', data_format = "channels_last",input_shape = image_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))

# Layer 2
model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = (1, 1), padding = 'valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))

# Layer 3
model.add(Conv2D(filters = 128, kernel_size=(3,3), strides=(1, 1), padding='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))
    

# Layer 4
model.add(Flatten(data_format = "channels_last"))
model.add(Dense(512))
model.add(Activation("relu"))

# Layer 5
model.add(Dense(256))
model.add(Activation("relu"))

# Layer 6
model.add(Dense(10,activation="softmax"))

sgd_optimizers = SGD(lr=learning_rate,decay=learning_rate_decay)

model.compile(optimizer = sgd_optimizers, loss=['categorical_crossentropy'], metrics=['accuracy'])
```

# Training 
Training is the process of feeding the data to neural network and modifiying the weights of the model using the the backpropagation algorithm. The backpropagation using loss the function acquires the loss over batch size of data and does a backpropagation to modify the weights in such a way the in the next epoch the loss would be less when compared to the current epoch


```python
# Training the model
model_history = model.fit(x=x_train, y=y_train_sparse, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test_sparse), shuffle=True)
```

    Train on 50000 samples, validate on 10000 samples
    Epoch 1/20
    50000/50000 [==============================] - 27s 539us/step - loss: 1.9462 - acc: 0.2773 - val_loss: 1.6804 - val_acc: 0.4021
    Epoch 2/20
    50000/50000 [==============================] - 24s 478us/step - loss: 1.5342 - acc: 0.4409 - val_loss: 1.3881 - val_acc: 0.4940
    Epoch 3/20
    50000/50000 [==============================] - 24s 475us/step - loss: 1.3288 - acc: 0.5199 - val_loss: 1.1933 - val_acc: 0.5803
    Epoch 4/20
    50000/50000 [==============================] - 24s 479us/step - loss: 1.2050 - acc: 0.5699 - val_loss: 1.0974 - val_acc: 0.6135
    Epoch 5/20
    50000/50000 [==============================] - 24s 475us/step - loss: 1.1114 - acc: 0.6036 - val_loss: 1.1717 - val_acc: 0.5881
    Epoch 6/20
    50000/50000 [==============================] - 24s 474us/step - loss: 1.0365 - acc: 0.6334 - val_loss: 0.9642 - val_acc: 0.6612
    Epoch 7/20
    50000/50000 [==============================] - 24s 476us/step - loss: 0.9716 - acc: 0.6572 - val_loss: 0.9205 - val_acc: 0.6759
    Epoch 8/20
    50000/50000 [==============================] - 24s 478us/step - loss: 0.9186 - acc: 0.6746 - val_loss: 0.8532 - val_acc: 0.6997
    Epoch 9/20
    50000/50000 [==============================] - 24s 477us/step - loss: 0.8761 - acc: 0.6905 - val_loss: 0.9445 - val_acc: 0.6698
    Epoch 10/20
    50000/50000 [==============================] - 24s 477us/step - loss: 0.8379 - acc: 0.7039 - val_loss: 0.8160 - val_acc: 0.7180
    Epoch 11/20
    50000/50000 [==============================] - 24s 477us/step - loss: 0.8057 - acc: 0.7165 - val_loss: 0.8235 - val_acc: 0.7112
    Epoch 12/20
    50000/50000 [==============================] - 24s 477us/step - loss: 0.7790 - acc: 0.7278 - val_loss: 0.7541 - val_acc: 0.7397
    Epoch 13/20
    50000/50000 [==============================] - 24s 477us/step - loss: 0.7513 - acc: 0.7347 - val_loss: 0.7825 - val_acc: 0.7277
    Epoch 14/20
    50000/50000 [==============================] - 24s 477us/step - loss: 0.7247 - acc: 0.7439 - val_loss: 0.7353 - val_acc: 0.7458
    Epoch 15/20
    50000/50000 [==============================] - 24s 477us/step - loss: 0.7019 - acc: 0.7517 - val_loss: 0.7442 - val_acc: 0.7412
    Epoch 16/20
    50000/50000 [==============================] - 24s 478us/step - loss: 0.6821 - acc: 0.7590 - val_loss: 0.7694 - val_acc: 0.7340
    Epoch 17/20
    50000/50000 [==============================] - 24s 477us/step - loss: 0.6621 - acc: 0.7683 - val_loss: 0.7090 - val_acc: 0.7581
    Epoch 18/20
    50000/50000 [==============================] - 24s 479us/step - loss: 0.6453 - acc: 0.7723 - val_loss: 0.7067 - val_acc: 0.7612
    Epoch 19/20
    50000/50000 [==============================] - 24s 478us/step - loss: 0.6269 - acc: 0.7788 - val_loss: 0.6944 - val_acc: 0.7600
    Epoch 20/20
    50000/50000 [==============================] - 24s 478us/step - loss: 0.6102 - acc: 0.7826 - val_loss: 0.6812 - val_acc: 0.7692
    

# Results
using the trained model we try to predict what are the values of images in the test set


```python
y_pred = model.predict(x=x_test, batch_size=batch_size, verbose=1)
```

    10000/10000 [==============================] - 2s 211us/step
    

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

    Ground truths of first 10 images in test set [[3]
     [8]
     [8]
     [0]
     [6]
     [6]
     [1]
     [6]
     [3]
     [1]]
    Predicted values of first 10 image in test set [3 8 8 0 6 6 1 6 3 1]
    


![png](/assets/images/CIFAR10_small_image_classification_keras_dataset_files/CIFAR10_small_image_classification_keras_dataset_19_1.png)



![png](/assets/images/CIFAR10_small_image_classification_keras_dataset_files/CIFAR10_small_image_classification_keras_dataset_19_2.png)


# Visulizing the results
checking the results by visulizing them and creating a confusion matrix. The values of precession and accuracy can be obtained by the help of confusion matrix and f1 scores to compare this architecure with other architectures of neural networks


```python
# Visulizing the results
y_pred = np.argmax(y_pred,axis=1)
y_test = y_test.ravel()
print("The shape of y_pred is ",np.shape( y_pred))
print("The shape of y_test is ",np.shape(y_test))
y_pred = pd.Series(y_pred,name = "predicted")
y_test = pd.Series(y_test,name = "Actual")
df_confusion  = pd.crosstab(y_test,y_pred)
df_confusion.columns = [i for i in list(cifar10_labels_dict.values())]
df_confusion.index = [i for i in list(cifar10_labels_dict.values())]

print(df_confusion)
plt.figure(figsize = (20,20))
plt.title('Confusion Matrix',fontsize=20)
sns.heatmap(df_confusion, annot=True,fmt="d")
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actaul', fontsize=18)
```

    The shape of y_pred is  (10000,)
    The shape of y_test is  (10000,)
                airplane  automobile  bird  cat  deer  dog  frog  horse  ship  \
    airplane         773          33    48    9    16    1    14     17    67   
    automobile         5         912     6    9     1    0    16      3    15   
    bird              48           7   706   43    60   24    74     24     8   
    cat               14          10    92  582    55   98    91     38    11   
    deer              12           3    79   38   727    8    72     53     7   
    dog               10           5    62  171    50  603    36     57     5   
    frog               3           2    45   35    19    4   884      2     4   
    horse             10           3    35   36    48   24    13    825     0   
    ship              34          33    13    9    11    3     4      4   874   
    truck             19          93    11    8     6    2    11     18    26   
    
                truck  
    airplane       22  
    automobile     33  
    bird            6  
    cat             9  
    deer            1  
    dog             1  
    frog            2  
    horse           6  
    ship           15  
    truck         806  
    




    Text(159.0, 0.5, 'Actaul')




![png](/assets/images/CIFAR10_small_image_classification_keras_dataset_files/CIFAR10_small_image_classification_keras_dataset_21_2.png)

