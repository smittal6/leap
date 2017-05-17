import sys
import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import htkmfc as htk

DIREC=["/home/smittal/Desktop/coding/leap/siddharth/kurtosis/train/single/",
       "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/train/overlap/",
       "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/test/single/",
       "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/test/overlap/"
      ]
FILES=["/home/smittal/Desktop/coding/leap/siddharth/kurtosis/kurtsingletrain.list",
       "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/kurtoverlaptrain.list",
       "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/kurtsingletest.list",
       "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/kurtoverlaptest.list"
      ]
savedirec="/home/smittal/Desktop/coding/leap/siddharth/kurtosis/"
trainfile="alltrainkurt"
testfile="alltestfile"

for i in range(0,4):
    print "Starting with: ",i," of 4"
    files=FILES[i]
    direc=DIREC[i]
    f=open(files)
    f=f.read()
    f=f.strip()
    f=re.split('\n',f)
    """
    if(i==0||i==2):
        index=0
    else:
        index=1
    """
    for i in range(len(f)): 
