import sys
import math
import wave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.fftpack
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

"""TEST REGION
# output=wave.open('briantest.wav','w')
# output.setnchannels(1)
# output.setsampwidth(2)
# output.setframerate(44100)
# output.writeframes(signal)
#Converting from string to integers. Required if we use wave library
# signal=np.fromstring(signal,'Int16')
# print signal.shape
#raw has the wave_read object and signal is the numpy array for values
"""
### Plotting the signal, using numpy array
plt.title('Sample plotting')
plt.plot(signal)
plt.savefig('beginner.png')
plt.close()

### Section for calculations, like FFT and MFCC

#some definitions
duration=2.5e-2 #25ms is defined to be the duration for FFT
shift_interval=duration/2 #duration/2
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

### WARNING: The following section is not correct. It takes moer frames.
### Please see the kurtosis code, for exact windowing.
for iterator in range(0,columns):
    vector_for_fft=signal[iterator*skip_entries:min(total_frames,samples+skip_entries*iterator)]
    hammer_size=vector_for_fft.shape[0]
    hamming_vector=np.hamming(hammer_size)
    hammed_vector=np.multiply(hamming_vector,vector_for_fft)
    fft_vector=np.fft.fft(hammed_vector,nfft)
    fft_matrix[:,iterator]=fft_vector

#Checking the matrix shape that we have
# print fft_matrix.shape 

#Because we are getting inverted matrix, we will take the mirror image about x axis
fft_matrix_useful=np.flipud(abs(fft_matrix[0:int(nfft/2)+1,:]))
### PLOTTING SPECTOGRAM
xlim=columns*duration/2
ylim=sampling_rate/2
xtics=np.linspace(0,xlim,columns)
ytics=np.linspace(0,ylim,int(nfft/2)+1)
plt.clf() #clearing the previous plot
np.set_printoptions(precision=2)
plt.imshow(20*np.log10(fft_matrix_useful),cmap="binary",aspect="auto",extent=[0,xlim,0,ylim]) #for complex values, abs returns the mag of the number
plt.title("Spectogram")
plt.xlabel("Time from Start")
plt.ylabel("Frequency [Hz]")
plt.colorbar()
plt.savefig('spectrogram.png')
plt.clf()
### Done with plotting

### MEL SECTION
#some parameters used in calling the functions
num_filters=40
# Some crucial function definitions
def hz2mel(hz):
    return 2595 * np.log10(1+hz/700.)
def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)
    # The following function is taken from python_speech_features implemented by James Lyons
def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)
    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

power_spectrum=np.square(abs(fft_matrix_useful)) # power_spectrum will have the powers for each frame.

"""
Structure of FFT Matrix: FFT Bin X Frames
Structure of FilterBank Matrix.T: FFT Bin X Filters
"""
filtered_spectrum=np.dot(np.flipud(fft_matrix_useful).T,get_filterbanks(num_filters,nfft,sampling_rate).T)
"""
Structure of filtered_spectrum: Frames X Filters.
That is for each frame(a row) we have entry in corresponding filters. Higher the entry in filter, higher the energy in that filter. We take logarithm of all the entries in the next stage
"""
log_spectrum=np.log(filtered_spectrum) #all entries for a frame are in row, thus column has filters
print "Log Spectrum Shape: ",log_spectrum.shape
plt.imshow(log_spectrum,aspect="auto",cmap="binary")
plt.title("Log Spectrum")
plt.colorbar()
plt.savefig("logspectrum.png")
plt.close()
dct_spectrum=scipy.fftpack.dct(log_spectrum,axis=1)
print "DCT Spectrum Shape: ",dct_spectrum.shape
# print dct_spectrum

### PLOT DCT MATRIX TO SEE THE VALUES OF THE COEFFICIENTS
plt.clf()
plt.imshow((dct_spectrum[:,1:10]),aspect="auto",cmap="binary")
plt.title("DCT SPECTRUM")
plt.colorbar()
plt.savefig("dctspectrum.png")

# It makes sense to take the first 13 DCT's, as the rest die out.
mfcc=dct_spectrum[:,0:13] #That is we are taking all the rows(ie frames) and than taking the first 14 coefficients
# np.set_printoptions(threshold='nan')
# print mfcc
