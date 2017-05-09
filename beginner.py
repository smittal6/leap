import sys
import pylab
import wave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#getting the wave_read object
raw=wave.open('brian.wav','r')
signal=raw.readframes(-1)
signal=np.fromstring(signal,'Int16')

#printing the signal
#plotting section
plt.title('Sample plotting')
plt.plot(signal)
plt.savefig('beginner.png')
