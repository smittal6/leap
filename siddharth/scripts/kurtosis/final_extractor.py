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
import re
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
          "/home/smittal/Desktop/coding/leap/siddharth/kurtosis/test/single"
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
    for i in range(len(f)):
        raw=wave.open(direc+f[i],'r')
        nchannels,sampwidth,sampling_rate,total_frames,comptype,compname=raw.getparams()
        sampling_rate,data=sciwav.read(direc+f[i])
        # print "The size of the raw data: ",data.shape
        print i," of ",len(f)
        if nchannels==1:
            signal=data
        else:
            signal=data[:,0]
        #some definitions
        duration=2.5e-2 #25ms is defined to be the duration for FFT
        shift_interval=1.0e-2 #duration/2
        samples=int(sampling_rate*duration) #These are the number of array entries that we'll use to find the kurtosis
        skip_entries=int(sampling_rate*shift_interval) #These entries are going to be skipped, that is we'll move to the next frame byb leaving these entries
        columns=int(math.ceil(total_frames/skip_entries))
        kurt_vals=np.empty(columns)
        signal=skp.normalize(signal)
        for iterator in range(0,columns):
            vector_for_kurt=signal[0][iterator*skip_entries:min(total_frames,samples+skip_entries*iterator)]
            kurt_vals[iterator]=stats.kurtosis(vector_for_kurt)
        kurt_vals=np.asarray(kurt_vals)
        sio.savemat(savdirec+re.split('[*.*]',f[i])[0]+'.mat',{'kurt':kurt_vals})
    print "Finish i=",i
