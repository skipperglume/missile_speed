import os
import scipy.io.wavfile as wavfile
def CutSound(input_name, output_name, time_period = 4000):
    if(input_name.endswith(".wav")):
        Fs, aud = wavfile.read(input_name)  
    else: 
        print("ERROR: NOT WAV FILE!")
        return -1
    print("Frame Rate: ", Fs)
    
    seconds = len(aud)/Fs
    minutes = len(aud)/Fs//60
    hours = len(aud)/Fs//60//60
    

    print("Total Length: ",hours,"hrs",minutes - hours*60,"mins",seconds- hours*60*60 - (minutes - hours*60)*60,"s" )
    print("In seconds: ", len(aud)/Fs)
    n_samples = int((len(aud)//Fs)//int(time_period//1000) )
    print("Total number of samples is: ",n_samples)
    
   # Comment
    
    for i in range(n_samples):
        filename = output_name+"/"+str(i)+"_.wav"
        wavfile.write(filename, Fs, aud[i*Fs*int(time_period//1000):((i+1)*Fs)*int(time_period//1000)])
    return -1