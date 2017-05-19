import numpy as np
import htkmfc as htk
import scipy.io as sio
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
# from keras.layers.convolutional import Convolution2D
# from keras.layers.convolutional import MaxPooling2D

import ConfigParser
import logging
import time
import sys
import os

np.random.seed(1337)
#trainfile="/home/smittal/Desktop/coding/leap/siddharth/combhtk/alltrain.htk"
#testfile="/home/smittal/Desktop/coding/leap/siddharth/combhtk/alltest.htk"
trainfile="/home/neerajs/siddharth/combhtk/alltrain.htk"
testfile="/home/neerajs/siddharth/combhtk/alltest.htk"
def load_data_train(trainfile):
    a=htk.open(trainfile)
    data=a.getall()
    #x_train=data[:,0:data.shape[1]-1]
    x_train=data[:,0:13]
    y_train=data[:,-1]
    return x_train,y_train
def load_data_test(testfile):
    a=htk.open(testfile)
    data=a.getall()
    #x_test=data[:,0:data.shape[1]-1]
    x_test=data[:,0:13]
    y_test=data[:,-1]
    return x_test,y_test
def seq(x_train,y_train,x_test,y_test):
    #Defining the structure of the neural network
    #Creating a Network, with 2 hidden layers.
    model=Sequential()
    model.add(Dense(256,activation='relu',input_dim=(x_train.shape[1]))) #Hidden layer1
    model.add(Dense(256,activation='relu')) #Hidden layer 2
    model.add(Dense(1,activation='sigmoid')) #Output Layer
    #Compilation region: Define optimizer, cost function, and the metric?
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    #Fitting region:Get to fit the model, with training data
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch)
    score=model.evaluate(x_test,y_test,batch_size=batch)
    print "\n"
    print "Score for the model: ",score
    scores=model.predict(x_test,batch_size=batch)
    #print scores.shape
    #print scores[1:5]
    sio.savemat(direc+name,{'scores':scores,'ytest':y_test})
#Non-function section
x_train,y_train=load_data_train(trainfile)
print "Loading training data complete"
print "Shape test: ",x_train.shape," ",y_train.shape
x_test,y_test=load_data_test(testfile)
print "Loading test data complete"
print "Shape test: ",x_test.shape," ",y_test.shape

#Some parameters for training the model
epochs=10 #Number of iterations to be run on the model while training
batch=128 #Batch size to be used while training
direc="/home/neerajs/siddharth/matrices/"
name="2dnn-mfcc-nodelta"
#Calling the seq model, with 2 hidden layers
seq(x_train,y_train,x_test,y_test)
