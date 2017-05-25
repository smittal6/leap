import os
import sys
import htkmfc as htk
import numpy as np

# Takes as input, The directory where the combined htk files are stored.
#And takes the name of the file, in which the final train data and final test data would be stored.
direct='/home/smittal/Desktop/coding/leap/siddharth/combhtk/mfcc-d-a-kurtosis-sfm/'
traindatafile=sys.argv[1] #Final all training data
testdatafile=sys.argv[2] #Final all test data
veclen=41 #Vector length,because of 39 features, and 1 from kurtosis, 1 from sfm

os.chdir(direct)

#SECTION FOR TRAIN DATA CREATION

a=htk.open('train_mfcc_k_sfm_s.htk')
data=a.getall()
b=htk.open('train_mfcc_k_sfm_s_labels.htk')
data=np.hstack((data,b.getall()))
print(data.shape)
a=htk.open('train_mfcc_k_sfm_o.htk')
data2=a.getall()
b=htk.open('train_mfcc_k_sfm_o_labels.htk')
data2=np.hstack((data2,b.getall()))
print(data2.shape)

Data=np.vstack((data,data2))
np.random.shuffle(Data)
writer=htk.open(traindatafile,mode='w',veclen=veclen+1)
writer=htk.open(traindatafile,mode='w',veclen=veclen+1)
writer.writeall(Data)

##SECTION FOR TEST DATA CREATION
a=htk.open('test_mfcc_k_sfm_s.htk')
data=a.getall()
b=htk.open('test_mfcc_k_sfm_s_labels.htk')
data=np.hstack((data,b.getall()))
print(data.shape)
a=htk.open('test_mfcc_k_sfm_o.htk')
data2=a.getall()
b=htk.open('test_mfcc_k_sfm_o_labels.htk')
data2=np.hstack((data2,b.getall()))
print(data2.shape)

Data=np.vstack((data,data2))
np.random.shuffle(Data)
writer=htk.open(testdatafile,mode='w',veclen=veclen+1)
writer=htk.open(testdatafile,mode='w',veclen=veclen+1)
writer.writeall(Data)
