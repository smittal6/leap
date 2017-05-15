import os
import sys
import htkmfc as htk
import numpy as np

#Takes as input, The directory where the combined htk files are stored.
#And takes the name of the file, in which the final train data and final test data would be stored.
direct=sys.argv[1]
traindatafile=sys.argv[2]
testdatafile=sys.argv[3]
veclen=39

os.chdir(direct)

#SECTION FOR TRAIN DATA CREATION
a=htk.open('single_train_data.htk')
data=a.getall()
b=htk.open('single_train_datalabels.htk')
data=np.hstack((data,b.getall()))
print(data.shape)
a=htk.open('overlap_train_data.htk')
data2=a.getall()
b=htk.open('overlap_train_datalabels.htk')
data2=np.hstack((data2,b.getall()))
print(data2.shape)

Data=np.vstack((data,data2))
np.random.shuffle(Data)
writer=htk.open(traindatafile,mode='w',veclen=veclen+1)
writer=htk.open(traindatafile,mode='w',veclen=veclen+1)
writer.writeall(Data)

##SECTION FOR TEST DATA CREATION
a=htk.open('single_test_data.htk')
data=a.getall()
b=htk.open('single_test_datalabels.htk')
data=np.hstack((data,b.getall()))
print(data.shape)
a=htk.open('overlap_test_data.htk')
data2=a.getall()
b=htk.open('overlap_test_datalabels.htk')
data2=np.hstack((data2,b.getall()))
print(data2.shape)

Data=np.vstack((data,data2))
np.random.shuffle(Data)
writer=htk.open(testdatafile,mode='w',veclen=veclen+1)
writer=htk.open(testdatafile,mode='w',veclen=veclen+1)
writer.writeall(Data)
