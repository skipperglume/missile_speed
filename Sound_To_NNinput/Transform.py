import os
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from   librosa import amplitude_to_db
from   librosa import stft
import librosa.display 
import numpy as np
import librosa

def Transform_Wav(input_name, output_name):
	
	# if(input_name.endswith(".wav")):
	
	# else: 
	#     print("ERROR: NOT WAV FILE!")
	#     return -1

	y, sr = librosa.load(input_name)
	print(sr)
	print(len(y))
	print(len(y)//sr)
	# D = stft(y, n_fft=256, hop_length=  sr*4//100+1)
	hop_length = sr*4//100+1
	n_fft = 198
	D = stft(y, n_fft=n_fft, hop_length= hop_length)
	print(input_name)
	print(D.shape) 
	
	S_db = amplitude_to_db(np.abs(D), ref=np.max)
  
	fig, ax = plt.subplots()
	img = librosa.display.specshow(S_db, ax=ax,sr=sr, hop_length=hop_length)
	fig.colorbar(img, ax=ax)
	fig, ax = plt.subplots()
	img = librosa.display.specshow(S_db, x_axis='time',sr=sr, hop_length=hop_length, y_axis='linear', ax=ax)
	ax.set(title='Now with labeled axes!')
	fig.colorbar(img, ax=ax, format="%+2.f dB")
	
	# librosa.stft.freq

	fig.savefig("plot.png")
	

	# print(librosa.fft_frequencies(sr, n_fft=n_fft))
	print(sr)
	# print(S_db[0])
	print(S_db.shape)
	
	return -1
def Transform_For_Input(input_name):
	# print("+I+")
	y, sr = librosa.load(input_name)
	if len(y)/sr>4:
		y = y[0:4*sr]
	if len(y)/sr<4:
		len_y = len(y[0])
		a = [0 for i in range(len_y)]
		b = [a for i in range(4*sr - len(y)+1)]
		y=y+b
		y = y[0:4*sr]
	if len(y)/sr == 4:
		y = y[0:4*sr]

	hop_length = sr*4//100+1
	n_fft = 198
	D = stft(y, n_fft=n_fft, hop_length= hop_length)
	# S_db = amplitude_to_db(np.abs(D), ref=np.max)
	# print(np.abs(D))
	# print(D.shape)
	# print(S_db.shape)
	D = np.abs(D)
	D/=np.max(D)
	return D



