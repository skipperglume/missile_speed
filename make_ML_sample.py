import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk


rocket_sound1 = *filepath*
scale, sr = librosa.load(rocket_sound1)
Signal = np.abs( librosa.stft(scale, hop_length = 890,n_fft=299))[20:120]
maxval= np.amax(Signal)

ouputarray=Signal/maxval
