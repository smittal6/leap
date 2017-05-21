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
import scipy.stats.mstats as stat
import librosa.util
import re
import htkmfc as htk
from scipy import stats
from scipy import signal

#first overlap train, then overlap test, then single train, single test

#These are the raw lists, which essentially contain the names of wav files.
FILES=["/home/smittal/Desktop/coding/leap/siddharth/lists/overlap_train_raw.list",
       "/home/smittal/Desktop/coding/leap/siddharth/lists/overlaptest/overtestraw.list",
       "/home/smittal/Desktop/coding/leap/siddharth/lists/raw_input.list",
       "/home/smittal/Desktop/coding/leap/siddharth/lists/singletest/singletestraw.list"
      ]
#We store the directories where raw audio files are stored, so that we can process them, and extract the feature that we want. if it is not MFCC
DIREC=["/home/smittal/Desktop/coding/leap/siddharth/overlap/train/",
       "/home/smittal/Desktop/coding/leap/siddharth/overlap/test/",
       "/home/smittal/Desktop/coding/leap/siddharth/WAV/wav/",
       "/home/smittal/Desktop/coding/leap/siddharth/WAV/wav_test/"
      ]

#Save the features extracted from this script, in the directory stated.
SAVDIREC=["/home/smittal/Desktop/coding/leap/siddharth/sfm/train/overlap/",
          "/home/smittal/Desktop/coding/leap/siddharth/sfm/test/overlap/",
          "/home/smittal/Desktop/coding/leap/siddharth/sfm/train/single/",
          "/home/smittal/Desktop/coding/leap/siddharth/sfm/test/single/"
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
        shift_interval=1.0e-2 #duration of shift
        samples=int(math.ceil(sampling_rate*duration)) #These are the number of array entries that we'll use to find the kurtosis
        skip_entries=int(math.ceil(sampling_rate*shift_interval)) #These entries are going to be skipped, that is we'll move to the next frame byb leaving these entries
        # columns=int(math.ceil(total_frames/skip_entries))
        sfm_vals=[]
        signal=skp.normalize(signal) #Normalization shouldn't have an effect on flatness measure
        iterator=0 #Just an iterator to control start and end points
        length=0 #Keeps track of the length that we have covered
        frames=0 #Keeps track of the number of frames
        while length<total_frames:
            start=iterator*skip_entries
            end=samples+start
            if end<total_frames:
                vector_for_sfm=signal[0][start:end]
                # sfm_vals.append(stats.kurtosis(vector_for_kurt,fisher=False))
                [freqs,psd]=signal.periodogram(vector_for_sfm,512) #512 point fft, as our signals are usually sampled at 16K and we use 25 ms window size
                sfm=stat.gmean(psd)/(np.mean(psd)+eps)
                sfm_vals.append(sfm)
                length=end
            else:
                vector_for_sfm=signal[0][start:total_frames]
                # sfm_vals.append(stats.kurtosis(vector_for_kurt,fisher=False))
                [freqs,psd]=signal.periodogram(vector_for_sfm,512)
                sfm=stat.gmean(psd)/(np.mean(psd)+eps)
                sfm_vals.append(sfm)
                length=total_frames
            iterator+=1
            frames+=1
        #Done with the loop
        sfm_vals=np.asarray(sfm_vals)
        sfm_val=sfm_vals[0:frames-1]
        # wri=htk.open(savdirec+re.split("[*.*]",f[j])[0]+'.htk',mode='w',veclen=frames-1)
        # print kurt_vals.shape
        # wri.writeall(kurt_val)
        sio.savemat(savdirec+re.split('[*.*]',f[j])[0]+'.mat',{'sfm':sfm_val})
    print "Finish i=",i
