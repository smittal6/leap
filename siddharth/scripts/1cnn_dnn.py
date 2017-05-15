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
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten

import ConfigParser
import logging
import time
import sys
import os

np.random.seed(1337)

Config = ConfigParser.ConfigParser()
Config.read(sys.argv[1])

def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

keyword = ConfigSectionMap("dataset")["keyword"]
frame = ConfigSectionMap("dataset")["frame"]
sample = ConfigSectionMap("dataset")["sample"]

nb_layers = "1"
batch_size = int(ConfigSectionMap("main")["batchsize"])
activation_func = ConfigSectionMap("main")["activation"]
lr= float(ConfigSectionMap("main")["learningrate"])
hidden_units= int(ConfigSectionMap("main")["hiddenunits"])
dropout= float(ConfigSectionMap("main")["dropout"])

nb_filters = int(ConfigSectionMap("convolution")["nb_filters"])
filter_size_x = int(ConfigSectionMap("convolution")["filter_size_x"])
filter_size_y = int(ConfigSectionMap("convolution")["filter_size_y"])
pool_x = int(ConfigSectionMap("convolution")["pool_x"])
pool_y = int(ConfigSectionMap("convolution")["pool_y"])
stride_x = int(ConfigSectionMap("convolution")["stride_x"])
stride_y = int(ConfigSectionMap("convolution")["stride_y"])

nb_neg_cycles = 3
nb_classes = 2

file_name = 'cnn_dnn_'+'nb'+str(nb_filters)+'_'+str(filter_size_x) + 'x' +str(filter_size_y)+ '_' +str(pool_x) + 'x' +str(pool_y)+ '_' +str(stride_x) +'x' +str(stride_y) + str(hidden_units)+'_'+str(dropout) 
log_file_name = file_name + ".log"
model_file_name = file_name + ".json"
weights_file_name = file_name + ".h5"

directory = keyword
log = "log"
posteriors = "posteriors"
weights = "weights"

if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists(directory+'/'+log):
    os.makedirs(directory+'/'+log)

if not os.path.exists(directory+'/'+posteriors):
    os.makedirs(directory+'/'+posteriors)

if not os.path.exists(directory+'/'+weights):
    os.makedirs(directory+'/'+weights)

path_to_dir = os.getenv("HOME")+"/Desktop/kws"

dataset_train = path_to_dir + '/code/data_generation/'+ keyword+'_'+frame+'_'+sample+'/'+keyword+'_train_data.htk'
dataset_val = path_to_dir + '/code/data_generation/'+ keyword+'_'+frame+'_'+sample+'/'+keyword+'_dev_data.htk'
dataset_dev = path_to_dir + '/code/data_generation/'+ keyword+'_'+frame+'_'+sample+'/'+keyword+'_test2_data.htk'
dataset_dev1 = path_to_dir + '/code/data_generation/'+ keyword+'_'+frame+'_'+sample+'/'+keyword+'_test1_data.htk'

dataset_noise = path_to_dir + '/code/data_generation/'+ keyword+'_'+frame+'_'+sample+'/'+keyword+'_noise_data.htk'


def load_data_train():
	print ("Loading Training Data")
	
	feats_reader=htk.open(dataset_train)
	train_data=feats_reader.getall()

	#Break the training set into X and Y
	y_train=train_data[:,-1]
	X_train=np.delete(train_data,-1,1)

	#Deleting the training data after val because we have created the X_train and y_train
	del train_data

	print ("Training Data : ",X_train.shape)
	print ("Training Output : ",y_train.shape)
	
	return (X_train,y_train)

def load_data_val(dataset):

	print ("Loading Validation Data")
	
	#Reading the data set.
	feats_reader=htk.open(dataset)
	val_data=feats_reader.getall()

	#Break the training set into X and Y
	y_val=val_data[:,-1]
	X_val=np.delete(val_data,-1,1)
	
	#Deleting the training data after val because we have created the X_train and y_train
	del val_data

	print ("Validation Data : ",X_val.shape)
	print ("Validation Output : ",y_val.shape)

	return (X_val,y_val)

def cnn_reshape(X_list):

	Y_list = []
	for i in X_list:
		j = i.reshape(int(frame),32)
		Y_list.append(j)
	Y_list = np.array(Y_list)
	Z_list = Y_list[:, np.newaxis, :, :]
	return(Z_list)

def class_accuracy(y_predicted, y_value, threshold=0.5 ):
#Calculates and returns the False Alarm Rate, False Reject Rate, True Alarm Rate, True Reject Rate.
	
	#Hypothesis
	false_reject = 0
	false_alarm = 0
	true_alarm = 0
	true_reject = 0

	#Total positive examples would be the sum of y_val because it would contain a 1 for every possible +ve example and 0 for -ve example
	total_positive_examples = sum(y_value)
	total_negative_examples = len(y_value) - total_positive_examples

	for i in range(0,len(y_predicted)):
		#Checking for the hypothesis
		if(y_predicted[i] >= threshold and y_value[i] == 0 ):
			false_alarm = false_alarm + 1
		elif(y_predicted[i] < threshold and y_value[i] == 1):
			false_reject = false_reject + 1
		elif(y_predicted[i] >= threshold and y_value[i] == 1):
			true_alarm = true_alarm + 1
		elif(y_predicted[i] < threshold and y_value[i] == 0):
			true_reject = true_reject + 1
		
	return (false_alarm/float(total_negative_examples), false_reject/float(total_positive_examples), true_alarm/float(total_positive_examples), true_reject/float(total_negative_examples))


def normalise_data(train_data, test_data):
#Normalises train and test data with respect to train data	
	
	#Normailising with respect to training data
	mean = np.mean(train_data,axis=0)
	var = np.var(train_data,axis=0)

	train_data = (train_data - mean)/np.sqrt(var)
	test_data = (test_data - mean)/np.sqrt(var)

	return (train_data, test_data, mean, var)


if __name__ == '__main__':
	
	logging.basicConfig(filename= directory+'/log/' + log_file_name, level=logging.INFO, format='%(message)s')
	logging.info("Parameters")

	with open(sys.argv[1]) as f:
		print(f)
		for line in f:
			logging.info(line)


	#Reading Data 
	(X_train, y_train)= load_data_train()
	(X_val, y_val) = load_data_val(dataset_val)
	

	X_train, X_val, mean, var = normalise_data(X_train, X_val)
	
	X_train = X_train.astype('float32')
	X_val = X_val.astype('float32')

	X_train = cnn_reshape(X_train)
	X_val = cnn_reshape(X_val)

	print(X_train.shape)
	print(X_val.shape)

	Y_train = np_utils.to_categorical(y_train,nb_classes)
	Y_val = np_utils.to_categorical(y_val,nb_classes)
	#The flag to keep the loop running
	run_flag=True
	weights=[]

	#Check if it is the first iteration
	first_iter=True

	#Setting the final accuracy to 0 just for the start
	final_acc=0.0
	count_neg_iter=0


	model = Sequential()
	model.add(Convolution2D(nb_filters, filter_size_x, filter_size_y, border_mode="valid", input_shape=(1, X_train.shape[2] , X_train.shape[3])))
	model.add(Activation(activation_func))
	model.add(MaxPooling2D(pool_size=(pool_x, pool_y), strides=(stride_x, stride_y)))
	model.add(Flatten())
	model.add(Dense(hidden_units))
	model.add(Activation(activation_func))
	model.add(Dropout(dropout))
	model.add(Dense(2))
	model.add(Activation('softmax'))
	y = model.summary()
	print(y)
	logging.info(y)


	iter_count=1
	 
	while run_flag :

		#Give ramdom weights in first iteration and previous weights to other
		if first_iter:
			first_iter=False
		else:
			model.set_weights(np.asarray(weights))
			
		logging.info("\n\nIteration:"+str(iter_count))
		print ("Learning Rate : ",lr)
		logging.info("Learning Rate: "+str(lr))


		#Definining the Stocastic Gradient Descent object
		sgd=SGD(lr=lr)
		#Compiling all the preset arguments
		model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
		#Training the model
		history = model.fit(X_train,Y_train, batch_size=batch_size, nb_epoch=1, verbose=1, validation_data=(X_val,Y_val))

		#Predicting the accuracy on validation data. This accuracy ensures weather it will run the loop again.
		Y_val_predicted = model.predict(X_val, verbose =1)
		Y_val_predicted_1 = Y_val_predicted[:,1]
		far,frr,tar,trr = class_accuracy(Y_val_predicted_1,y_val)
		total_positive_examples_val = sum(y_val)
		total_negative_examples_val = len(y_val) - total_positive_examples_val
		#Giving weights to positive vaildation set so as to ensure it runs well 
		accuracy = (3*total_positive_examples_val*tar + total_negative_examples_val * trr) / (3*total_positive_examples_val + total_negative_examples_val)
		current_acc = accuracy   

		#Comparing the current and final accuracy and if > 0.001 make that the final accuracy
		if(current_acc - final_acc > 0):		
			iter_count = iter_count + 1

			#Update the weights if the accuracy is greater than .001
			weights=model.get_weights()
			print ("Updating the weights")
			logging.info("Updating the weights")
			#Updating the final accuracy
			final_acc=current_acc
			#Setting the count to 0 again so that the loop doesn't stop before reducing the learning rate n times consecutivly
			count_neg_iter = 0

		else:
			#If the difference is not greater than 0.005 reduce the learning rate
			lr=lr/2.0
			print ("Reducing the learning rate by half")
			logging.info("Reducing the learning rate by half")
			count_neg_iter = count_neg_iter + 1
			
			#If the learning rate is reduced consecutively for nb_neg_cycles times then the loop should stop
			if(count_neg_iter>nb_neg_cycles):
				run_flag=False
				model.set_weights(np.asarray(weights))
				
				#Saving the model and weights in a separate file
				json_string = model.to_json()
				open(keyword+"/weights/"+model_file_name,'w').write(json_string)
				model.save_weights(keyword+"/weights/"+weights_file_name, overwrite=True)



	del X_train, X_val


	time.sleep(5)

        (X_dev, y_dev) = load_data_val(dataset_dev)
        (X_noise, y_noise) = load_data_val(dataset_noise)

        X_dev = (X_dev - mean)/np.sqrt(var)
        X_noise = (X_noise - mean)/np.sqrt(var)

        X_dev = X_dev.astype('float32')
        X_noise = X_noise.astype('float32')

        X_dev = cnn_reshape(X_dev)
        X_noise = cnn_reshape(X_noise)

        print('Development Data shape,', X_dev.shape)
        print('Noise Data shape,', X_noise.shape)

        Y_dev = np_utils.to_categorical(y_dev,nb_classes)
        Y_noise = np_utils.to_categorical(y_noise,nb_classes)


	#print ("Final Accuracy : ",final_acc)
        #logging.info("Final Accuracy: "+str(final_acc))

        #Predicting the value of X_val
        Y_predicted = model.predict(X_dev, verbose = 1)
        Y_predicted_noise = model.predict(X_noise, verbose = 1)

        #print (Y_predicted,'\n --- \n', Y_predicted_noise)
        #Since Y_predicted has 2 columns for the probability of 0 and 1. We should take the 2nd column i.e. the probability of one
        #Y_predicted_1 = Y_predicted[:,1]
        #Y_predicted_noise_1 = Y_predicted_noise[:,1]

        final_output=np.concatenate((Y_predicted, Y_dev), axis=1)
        final_output_noise = np.concatenate((Y_predicted_noise, Y_noise), axis = 1)

        np.savetxt(keyword + "/"+ posteriors + "/" + file_name +"posterior_test2.txt",final_output)
        np.savetxt(keyword + "/" + posteriors + "/" + file_name + "posterior_noise.txt",final_output_noise)


        del X_dev, X_noise

        time.sleep(5)

        (X_dev1, y_dev1) = load_data_val(dataset_dev1)

        X_dev1 = (X_dev1 - mean)/np.sqrt(var)
        X_dev1 = X_dev1.astype('float32')
        X_dev1 = cnn_reshape(X_dev1)
        print('Development Data shape,', X_dev1.shape)

        Y_dev1 = np_utils.to_categorical(y_dev1,nb_classes)
        Y_predicted1 = model.predict(X_dev1, verbose = 1)
        #Y_predicted_11 = Y_predicted1[:,1]

        final_output1=np.concatenate((Y_predicted1, Y_dev1), axis=1)
        np.savetxt(keyword + "/"+ posteriors + "/" + file_name +"posterior_test1.txt",final_output1)

	print("Successfully Completed! Peace")
