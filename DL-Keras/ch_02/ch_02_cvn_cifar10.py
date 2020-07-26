"""
Version: 0.1

Convolutional Network that learns to recnognize handwriting from the MINIST dataset

Results for this version:

    TODO: Testar locally

    Test score:     
    Test accuracy:  
    
"""


from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
IMG_CHANNELS = 3
IMG_ROWS     = 32
IMG_COLS     = 32

# constant
BATCH_SIZE   = 128
NB_EPOCH     = 20
NB_CLASSES   = 10
VERBOSE      = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# load dataset

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print ("X_train shape: ",X_train.shape)
print (X_train.shape[0],'train samples')
print (X_test.shape[0], 'test samples')

# convert categorical
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test,NB_CLASSES)

# float and normalization
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32') 
X_train /= 255
X_test  /= 255

model = Sequential()

"""
- 32 convolutional filters, with (3 x 3)
- Output dimension is the same one of the input shape (32 x 32)
"""
model.add(Conv2D(32,(3,3), padding='same', input_shape=(IMG_ROWS,IMG_COLS, IMG_CHANNELS)))
# Activation ReLU (simple way of introducing non-linearity)
model.add(Activation('relu'))
# Max-pooling operation with pool size of (2 x 2)
model.add(MaxPooling2D(pool_size=(2,2)))
# dropout at 25%
model.add(Dropout(0.25))

# Flattens the input
model.add(Flatten())
# Add a Dense layer with 512 units
model.add(Dense(512))
model.add(Activation('relu'))
# Dropout of 50%
model.add(Dropout(0.5))
# Add softmax layer with 10 classes as output
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# train
model.compile(loss='categorical_crossentropy',optimizer=OPTIM,
            metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
          validation_split=VALIDATION_SPLIT,verbose=VERBOSE)

score = model.evaluate(X_test, Y_test,batch_size=BATCH_SIZE,verbose=VERBOSE)

print ("Test score: ",score[0])
print ("Test score: ",score[1])

# save model
model_json = model.to_json()
open('cifar10_architecture_v2.json','w').write(model_json)

model.save_weights('cifar10_weights_v2.h5',overwrite=True)