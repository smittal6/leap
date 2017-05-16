import numpy as np
import htkmfc as htk

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
trainfile="/home/smittal/Desktop/coding/leap/siddharth/combhtk/alltest.htk"
testfile="/home/smittal/Desktop/coding/leap/siddharth/combhtk/alltrain.htk"
def load_data_train(trainfile):
    a=htk.open(trainfile)
    data=a.getall()
    x_train=data[:,0:data.shape[1]-1]
    y_train=data[:,-1]
    return x_train,y_train
def load_data_test(testfile):
    a=htk.open(testfile)
    data=a.getall()
    x_test=data[:,0:data.shape[1]-1]
    y_test=data[:,-1]
    return x_test,y_test
def seq(x_train,y_train,x_test,y_test):
    #Defining the structure of the neural network
    #Creating a Network, with 2 hidden layers.
    model=Sequential()
    model.add()
    model.add()

    #Compilation region: Define optimizer, cost function, and the metric?

    #Fitting region:Get to fit the model, with training data


#Non-function section
x_train,y_train=load_data_train(trainfile)
print "Loading training data complete"
print "Shape test: ",x_train.shape," ",y_train.shape
x_test,y_test=load_data_test(testfile)
print "Loading test data complete"
print "Shape test: ",x_test.shape," ",y_test.shape
