#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:50:50 2020

@author: alexjiang
"""

#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

f = open("simdata.csv", "r")
x = []

#loop through file by line break
for line in f:
	#cleaning up the line so that it is a list with first element as whether it is normal or uniform, second as the y value, and rest as x values
    temp = [float(i) for i  in (line.strip("\n").split(" "))]
    #adding the list to the larger data array
    x.append(temp)

#setting up training and test data and converting such data to numpy arrays
x_train = []
x_test = []
for i in range(len(x)):
    if (i%2 == 0):
        x_train.append(x[i])
    else:
        x_test.append(x[i])
x_train = np.asarray(x_train, dtype=np.float64) 
x_test = np.asarray(x_test, dtype=np.float64)
    
y_train = []
y_test = []
for i in x_train:
    y_train.append(i[0])
    np.delete(x_train, 0)
for i in x_test:
    y_test.append(i[0])
    np.delete(x_test, 0)
y_train = np.asarray(y_train, dtype=np.float64)
y_test = np.asarray(y_test, dtype=np.float64)

 

mnist = tf.keras.datasets.mnist
#below are the datasets to be used. x_train is a large list. 1 index in that is another list. 1 index in that is another list and within that are numbers
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
#print(type(x_train[0][0][0]))

#making of the model architecture
model = tf.keras.models.Sequential()
#input layer
model.add(tf.keras.layers.Flatten())
#first hidden layer, first parameter is neurons/units in layer. second input is sigmoid/activation function (what makes it fire)
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
#second layer
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
#output layer. #use softmax for probability distribution
model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))

#in case of cats and dogs instead of sparse do binary
#sparse_categorical_crossentropy
#optimizer can change to sigmoid
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


#training parameters
model.fit(x_train, y_train, epochs = 3)


#account for overfitting
val_loss, val_acc = model.evaluate(x_test, y_test)
#print(val_loss, val_acc)
#model.save('epic_num_reader.model')
#new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = model.predict([x_test])
#print(predictions)
#the 0th prediction in x_test
print(predictions[650])
print(np.argmax(predictions[650]))



#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()

f.close()

