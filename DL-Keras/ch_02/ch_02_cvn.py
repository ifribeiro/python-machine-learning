"""
Version: 0.1

Convolutional Network that learns to recnognize handwriting from the MINIST dataset

Results for this version:

    Test score:     0.04221847515421332
    Test accuracy:  0.9907000064849854

"""

from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

#Class definition
class LeNet:
  @staticmethod
  def build(input_shape, classes):
    model = Sequential()

    # CONV => RELU => POOL
    model.add(Conv2D(20,kernel_size=5,padding="same",input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Flatten => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # Softmax classifier
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model

NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28, 28 # Input image dimensions
NB_CLASSES = 10

"""
had to change this line, as my Keras instalation
does not have the K.set_image_dim_ordering("th") function
"""
INPUT_SHAPE = (IMG_ROWS, IMG_COLS,1)

# data: shuffled and split between train and test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# consider them as float and normalize
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_test /= 255

"""
We need a 60K x [28 x 28 x 1] shape as input to the CONVNET 
OBS:
    had to change the following 2 lines, as my Keras instalation
    does not have the K.set_image_dim_ordering("th") function
"""

X_train = X_train[:,:,:,np.newaxis]
X_test  = X_test [:,:,:,np.newaxis]
print (X_train.shape[0], "train samples")
print (X_test.shape[0], "test samples")

# Convert class vectors to binary class matrices
y_train  = np_utils.to_categorical(y_train, NB_CLASSES)
y_test   = np_utils.to_categorical(y_test, NB_CLASSES)

model = LeNet.build(input_shape=INPUT_SHAPE,classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy",optimizer=OPTIMIZER,metrics=["accuracy"])
history = model.fit(X_train, y_train,
            batch_size=BATCH_SIZE, epochs=NB_EPOCH,
            verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, y_test, verbose=VERBOSE)

print ("Test score:    ", score[0])
print ("Test accuracy: ", score[1])

# list all data history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel("accuracy")
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

# list all data history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel("loss")
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()