#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:19:52 2017

Goal: Biuld a VGGNet type model on Keras for image classification
Dataset : CIFAR10

@author: sabpap
"""
#ImportLibraries
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

#fix random seed
seed = 5
np.random.seed(5)

#load dataset
(x_train , y_train), (x_test, y_test) = cifar10.load_data()
print("CIFAR10 dataset loaded...")

### data preprocess ###

#normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

#subtracting the mean RGB 
m = x_train.mean(0)
x_train -= m
x_test -= m

#one-hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#define image generator for data augmentation(used if aug = 1)
imgen = ImageDataGenerator(featurewise_center=False,
                            rotation_range=40.,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            fill_mode='nearest',
                            )

###

### Build Model ###

in_shape = (32,32,3)

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=in_shape, padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(4)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(4)))
model.add(Dense(num_classes, activation='softmax'))
from time import time
from keras.callbacks import TensorBoard

# Compile Model

epochs = 100
lrate = 0.01
decay = lrate/epochs
batch_size = 32
aug = False

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


###

### Train Model ###


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard]) 

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=02)
print("Accuracy: %.2f%%" % (scores[1]*100))

if aug :
    print("Data augmentation Generator not ready.Not used")
###
    
### Save Model ###
    
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
###    