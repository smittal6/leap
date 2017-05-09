import sys
import math
import wave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import librosa as librosa
### Input section

#getting the wave_read object
raw=wave.open('myvoice.wav','r') #myvoice.wav has sampling rate of 16K Hz

#reading all the frames. -1 indicates the end of file.
signal=raw.readframes(-1)

#Converting from string to integers.
signal=np.fromstring(signal,'Int16')
np.set_printoptions(threshold='nan') #for printing the whole array, so that it doesn't truncate the array simply.

#raw has the wave_read object and signal is the numpy array for values

#printing the signal values, as an array and checking its linearity
#print signal
# print signal.shape

### Plotting the signal, using numpy array
plt.title('Sample plotting')
plt.plot(signal)
plt.savefig('beginner.png')
plt.close()
### Section for calculations, like FFT and MFCC

#some definitions
sampling_rate=raw.getframerate() #getting the samplin rate
total_frames=raw.getnframes() #gives the total number of frames
duration=2.5e-2 #25ms is defined to be the duration for FFT
shift_interval=1e-2 #10ms is defined to be the shift interval, for the windows.

#verifying the sampling rate
# print sampling_rate

samples=int(sampling_rate*duration) #These are the number of array entries that we'll use to find the FFT
skip_entries=int(sampling_rate*shift_interval) #These entries are going to be skipped.


print "Number of data points in a frame: ",samples #displaying how many data points we are going to take
print "Total data points: ",total_frames

#finding the number of rows and columns of the FFT matrix
#number of rows is essentially the dim of FFT space
#number of columns is the number of frames that are going to be captured.
length_transformed=int(pow(2,int(math.log(samples,2))+1))
columns=int(math.ceil(total_frames/samples))
fft_matrix=np.empty([length_transformed,columns],dtype=np.complex_) #defining the numpy object of required dim

print "Number of frames required: ",columns


#running the loop for finding the FFT Matrix
for iterator in range(0,columns):
    vector_for_fft=signal[iterator*skip_entries:min(total_frames,samples+skip_entries*iterator)]
    # print vector_for_fft
    hammer_size=vector_for_fft.shape[0]
    # print iterator,hammer_size
    hamming_vector=np.hamming(hammer_size)
    hammed_vector=np.multiply(hamming_vector,vector_for_fft)
    fft_vector=np.fft.fft(hammed_vector,length_transformed)
    fft_matrix[:,iterator]=fft_vector

#Checking the matrix shape that we have
# print fft_matrix.shape 

plt.clf() #clearing the previous plot
# plt.figure(figsize=(5,5))
plt.xlim(0,200)
plt.imshow(abs(fft_matrix),cmap="binary",aspect="auto") #for complex values, abs returns the magnitude of the number
# plt.title("First column of the FFT Matrix")
# plt.xlabel("FFT Space Point")
# plt.ylabel("FFT Coefficient")
# plt.show()
plt.savefig('spectrogram.png')
# print librosa.feature.mfcc(y=signal) #for checking the mfcc's
