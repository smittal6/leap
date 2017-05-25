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
    writer=htk.open(save_addr+filename+'.htk',mode='w',veclen=num)
    for i in range(len(file_reader)):
    # for i in range(100):
        print(i)
        raw_name=re.split("[*.*]",file_reader[i])[0]
        data_read=htk.open(addr+raw_name+'.htk')
        kurt_read=sio.loadmat(kurt_addr+raw_name+'.mat')['kurt']
        sfm_read=sio.loadmat(sfm_addr+raw_name+'.mat')['sfm']
        try:
            read_data=data_read.getall()
            print "MFCC data shape: ",read_data.shape
            # kurt_data=kurt_read.getall()
            print "Kurtosis vector shape: ",kurt_read.shape
            if read_data.shape[0] == kurt_read.shape[1]:
                fwritedata=np.hstack((read_data,np.transpose(kurt_read),np.transpose(sfm_read)))
            else:
                corrupt_files+=1
                print "The frames of kurtosis data and HTK data did not match"
                continue
        except:
            corrupt_files+=1
            continue
        ind=ind+read_data.shape[0]
        writer.writeall(fwritedata)
        print('Corrput files',corrupt_files)
    print('Corrput files',corrupt_files)
    labels=np.ones((ind,1))
    labels=labels*index
    print(labels.shape)
    wri=htk.open(save_addr+filename+'_labels.htk',mode='w',veclen=1)
    wri.writeall(labels)

##Correct the following paths
# if os.path.isdir('/home/smittal/Desktop/coding/leap/siddharth/combhtk/mfcc-d-a-kurtosis'):
    # pass
# else:
    # os.mkdir('/home/smittal/Desktop/coding/leap/siddharth/combhtk/mfcc-d-a-kurtosis')
    # os.chdir('/home/smittal/Desktop/coding/leap/siddharth/combhtk/mfcc-d-a-kurtosis')

addr='/home/smittal/Desktop/coding/leap/siddharth/feats/test/aftervado/'
kurt_addr='/home/smittal/Desktop/coding/leap/siddharth/kurtosis/test/aftervado/'
sfm_addr='/home/smittal/Desktop/coding/leap/siddharth/sfm/test/aftervado/'
save_addr='/home/smittal/Desktop/coding/leap/siddharth/combhtk/mfcc-d-a-kurtosis-sfm/'
num=39+1+1 #39 for mfcc,d,a and 1 for kurtosis of each frame
file_read='/home/smittal/Desktop/coding/leap/siddharth/lists/overlaptest/overtestraw.list'
index=1 #For single speaker data, index is 0
filename='test_mfcc_k_sfm_o'
file_reader=file_opener(file_read)
data_creator(num,addr,file_reader,index,filename)

