from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import Adam
from keras.utils import np_utils


# reading the dataset
df = pd.read_csv('graf_test.csv')

X = df[['C1','C2']]
Y = df['Cls']

SHAPE = len(X)

# spliting train and test 
X_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25)

# network and training

NB_EPOCH = 200
BATCH_SIZE = 128 # slices of array that will be used in each epoch
VERBOSE = 1
NB_CLASSES = 2
OPTIMIZER = Adam()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

X_train.astype('float32')
x_test.astype('float32')

# 2 outputs
model = Sequential()

# input shape (formato de cada entrada)
model.add(Dense(NB_CLASSES, input_shape=(2,)))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train,y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(x_test, y_test, verbose=VERBOSE)

print ("Teste score:", score[0])
print ("Teste accuracy:", score[1])