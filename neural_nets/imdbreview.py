
# coding: utf-8

# In[33]:

import keras
import numpy
import pydot_ng as pydot
import graphviz
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras import optimizers
from keras import losses

maxlen=500


# In[35]:


def load_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",num_words=None,skip_top=0,maxlen=None,seed=113,start_char=1,oov_char=2,index_from=3)
    return (x_train,y_train),(x_test,y_test)
def seq(x_train,y_train,x_test,y_test):
    #Defining the model, and its architecture
    model=Sequential()
    model.add(Dense(32,input_dim=maxlen,activation="relu" ))
    model.add(Dense(1,activation="sigmoid"))

    #Compiling the model, optimizer, loss function, metrics
    model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
    
    model.summary()
    #Now, fitting the model
    model.fit(x_train,y_train,epochs=1,batch_size=128)
    plot_model(model)
    score=model.evaluate(x_test,y_test)
    return score


# In[31]:


(x_train,y_train),(x_test,y_test)=load_data()

# print x_train
#The reviews have been preprocessed and have been encoded as a sequence of word indexes(integers). Say a word has index 4, then we encounter 4 in a review it means that the corresponding word was there in the review in its place
x_train=sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=sequence.pad_sequences(x_test,maxlen=maxlen)
# print x_train[0]


# In[36]:


print x_train.shape
print x_test.shape
print "Score for the model: ",seq(x_train,y_train,x_test,y_test)

