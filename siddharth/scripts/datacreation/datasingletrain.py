import re
import os
import sys
import time
import numpy as np
import scipy.io as sio
import htkmfc as htk

#The aim of this file is to create two htk files, one with all data matrix for one file, and other with all the labels

def file_opener(file_read):
        file_reader=open(file_read)
        file_reader=file_reader.read()
        file_reader=file_reader.strip()
        file_reader=re.split('\n',file_reader)
        return file_reader

def data_creator(num,addr,file_reader,index,filename):
    	corrupt_files=0
	ind=0
    	writer=htk.open(filename+'.htk',mode='w',veclen=num)
	for i in range(len(file_reader)):
        	print(i)
            	data_read=htk.open(addr+re.split("[*.*]",file_reader[i])[0]+'.htk')
            	try:
                	read_data=data_read.getall()
		except:
			corrupt_files+=1
			continue
		ind=ind+read_data.shape[0]
		writer.writeall(read_data)
	print('Corrput files',corrupt_files)
    	labels=np.ones((ind,1))
    	labels=labels*index
	print(labels.shape)
	wri=htk.open(filename+'labels.htk',mode='w',veclen=1)
	wri.writeall(labels)

if os.path.isdir('/home/neerajs/siddharth/combhtk'):
	pass
else:
	os.mkdir('/home/neerajs/siddharth/combhtk')
os.chdir('/home/neerajs/siddharth/combhtk')

addr='/home/neerajs/siddharth/train/mfcc/'
num=39
file_read='/home/neerajs/siddharth/lists/raw_input.list'
index=0
filename='single_train_data'
file_reader=file_opener(file_read)
data_creator(num,addr,file_reader,index,filename)

