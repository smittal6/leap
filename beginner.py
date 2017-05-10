import sys
import math
import wave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile as sciwav
### Input section

#getting the wave_read object
raw=wave.open(sys.argv[1],'r')
nchannels,sampwidth,sampling_rate,total_frames,comptype,compname=raw.getparams()
sampling_rate,data=sciwav.read(sys.argv[1])
print "The size of the raw data: ",data.shape
if nchannels==1:
    signal=data
else:
    signal=data[:,0] #taking one of the channels.
# signal=raw.readframes(-1)

###TEST REGION
# output=wave.open('briantest.wav','w')
# output.setnchannels(1)
# output.setsampwidth(2)
# output.setframerate(44100)
# output.writeframes(signal)
#Converting from string to integers. Required if we use wave library
# signal=np.fromstring(signal,'Int16')
# print signal.shape
#raw has the wave_read object and signal is the numpy array for values

### Plotting the signal, using numpy array
plt.title('Sample plotting')
plt.plot(signal)
plt.savefig('beginner.png')
plt.close()

### Section for calculations, like FFT and MFCC

#some definitions
duration=2.5e-2 #25ms is defined to be the duration for FFT
shift_interval=1e-2 #10ms is defined to be the shift interval, for the windows.
samples=int(sampling_rate*duration) #These are the number of array entries that we'll use to find the FFT
skip_entries=int(sampling_rate*shift_interval) #These entries are going to be skipped.


#number of rows is essentially the dim of FFT space
#number of columns is the number of frames that are going to be captured.
nfft=int(pow(2,int(math.log(samples,2))+1)) #These are the number of space points in FFT
#FFT is symmetric about Space points by 2, because of "2" factor that comes in the FFT formula.
columns=int(math.ceil(total_frames/skip_entries))
fft_matrix=np.empty([nfft,columns],dtype=np.complex_) #defining the numpy object of required dim


print "The sampling rate: ",sampling_rate
print "Number of data points in a frame: ",samples #displaying how many data points we are going to take
print "Total data points: ",total_frames #This denotes the number of total data collected, ie all the pressure variations recorded.
print "Number of frames required: ",columns


for iterator in range(0,columns):
    vector_for_fft=signal[iterator*skip_entries:min(total_frames,samples+skip_entries*iterator)]
    hammer_size=vector_for_fft.shape[0]
    hamming_vector=np.hamming(hammer_size)
    hammed_vector=np.multiply(hamming_vector,vector_for_fft)
    fft_vector=np.fft.fft(hammed_vector,nfft)
    fft_matrix[:,iterator]=fft_vector

#Checking the matrix shape that we have
# print fft_matrix.shape 

#creating the vector for mapping column to time, in the original signal.
xlim=columns*duration/2
ylim=sampling_rate/2
xtics=np.linspace(0,xlim,columns)
ytics=np.linspace(0,ylim,int(nfft/2))
plt.clf() #clearing the previous plot
# ax=plt.gca()
# ax.set_xticklabels(xtics)
np.set_printoptions(precision=2)
plt.imshow(20*np.log10(abs(fft_matrix[0:int(nfft/2),:])),cmap="binary",aspect="auto",extent=[0,xlim,0,ylim]) #for complex values, abs returns the mag of the number
plt.title("Spectogram")
plt.xlabel("Time from Start")
plt.ylabel("Frequency [Hz]")
plt.colorbar()
# plt.show()
plt.savefig('spectrogram.png')
