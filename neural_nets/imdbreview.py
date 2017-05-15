import keras
import numpy
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import losses

def load_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",num_words=None,skip_top=0,maxlen=None,seed=113,start_char=1,oov_char=2,index_from=3)
    return (x_train,y_train),(x_test,y_test)
def seq(x_train,y_train,x_test,y_test):
    #Defining the model, and its architecture
    model=Sequential([
    Dense(32,input_shape=(25000,)),
    Activation('relu'),
    Dense(10),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid')])

    #Compiling the model, optimizer, loss function, metrics
    model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

(x_train,y_train),(x_test,y_test)=load_data()
# print x_train

#The reviews have been preprocessed and have been encoded as a sequence of word indexes(integers). Say a word has index 4, then we encounter 4 in a review it means that the corresponding word was there in the review in its place
seq(x_train,y_train,x_test,y_test)
