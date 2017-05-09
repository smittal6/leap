import sys
import pylab
import wave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

### Input section

#getting the wave_read object
raw=wave.open('myvoice.wav','r') #myvoice.wav has sampling rate of 16K Hz

#reading all the frames. -1 indicates the end of file.
signal=raw.readframes(-1)

#Converting from string to integers.
signal=np.fromstring(signal,'Int16')
np.set_printoptions(threshold='nan') #for printing the whole array, so that it doesn't truncate the array simply.

#printing the signal values, as an array and checking its linearity
#print signal
# print signal.shape

### Plotting the signal, using numpy array
plt.title('Sample plotting')
plt.plot(signal)
plt.savefig('beginner.png')

### Section for calculations, like FFT and MFCS

#some definitions
sampling_rate=raw.getframerate() #getting the samplin rate
duration=2.5e-2 #25ms is defined to be the duration for FFT
shift_interval=1e-2 #10ms is defined to be the shift interval, for the windows.

#verifying the sampling rate
# print sampling_rate #verified. Correctly got the sampling rate.

