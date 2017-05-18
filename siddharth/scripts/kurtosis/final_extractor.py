import sys
import math
import wave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.io as sio
import scipy.io.wavfile as sciwav
import sklearn.preprocessing as skp
import librosa.util
import re
import htkmfc as htk
from scipy import stats

#first overlap train, then overlap test, then single train, single test

FILES=["/home/smittal/Desktop/coding/leap/siddharth/lists/overlap_train_raw.list",
       "/home/smittal/Desktop/coding/leap/siddharth/lists/overlaptest/overtestraw.list",
       "/home/smittal/Desktop/coding/leap/siddharth/lists/raw_input.list",
       "/home/smittal/Desktop/coding/leap/siddharth/lists/singletest/singletestraw.list"
      ]
DIREC=["/home/smittal/Desktop/coding/leap/siddharth/overlap/train/",
       "/home/smittal/Desktop/coding/leap/siddharth/overlap/test/",
       "/home/smittal/Desktop/coding/leap/siddharth/WAV/wav/",
       "/home/smittal/Desktop/coding/leap/siddharth/WAV/wav_test/"
      ]
SAVDIREC=["/home/smittal/Desktop/coding/leap/siddharth/kurtosis/train/overlap/",
          "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/test/overlap/",
          "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/train/single/",
          "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/test/single/"
         ]
for i in range(0,4):
    print "Starting with: ",i," of 4"
    files=FILES[i]
    direc=DIREC[i]
    savdirec=SAVDIREC[i]
    f=open(files)
    f=f.read()
    f=f.strip()
    f=re.split('\n',f)
    for j in range(len(f)):
        raw=wave.open(direc+f[j],'r')
        nchannels,sampwidth,sampling_rate,total_frames,comptype,compname=raw.getparams()
        sampling_rate,data=sciwav.read(direc+f[j])
        # print "The size of the raw data: ",data.shape
        #print i," of ",len(f)
        if nchannels==1:
            signal=data
        else:
            signal=data[:,0]
        #some definitions
        duration=2.5e-2 #25ms is defined to be the duration for FFT
        shift_interval=1.0e-2 #duration
        samples=int(math.floor(sampling_rate*duration)) #These are the number of array entries that we'll use to find the kurtosis
        skip_entries=int(math.floor(sampling_rate*shift_interval)) #These entries are going to be skipped, that is we'll move to the next frame byb leaving these entries
        # columns=int(math.ceil(total_frames/skip_entries))
        kurt_vals=[]
        signal=skp.normalize(signal)
        iterator=0 #Just an iterator to control start and end points
        length=0 #Keeps track of the length that we have covered
        frames=0 #Keeps track of the number of frames
        while length<total_frames-1:
            start=iterator*skip_entries
            end=samples+start
            if end<total_frames:
                vector_for_kurt=signal[0][start:end]
                kurt_vals.append(stats.kurtosis(vector_for_kurt,fisher=False))
                length=end
            else:
                vector_for_kurt=signal[0][start:total_frames]
                kurt_vals.append(stats.kurtosis(vector_for_kurt,fisher=False))
                length=total_frames
            iterator+=1
            frames+=1
        #Done with the loop
        kurt_vals=np.asarray(kurt_vals)
        kurt_val=kurt_vals[0:frames-1]
        # wri=htk.open(savdirec+re.split("[*.*]",f[j])[0]+'.htk',mode='w',veclen=frames-1)
        # print kurt_vals.shape
        # wri.writeall(kurt_val)
        sio.savemat(savdirec+re.split('[*.*]',f[j])[0]+'.mat',{'kurt':kurt_val})
    print "Finish i=",i
