import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def filter_df(df):
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, df)

def main():
    data = pd.read_csv('Data_summary.csv')
    names = ['1_left', '1_right', '2_left', '2_right', '3_left', '3_right', '4_left', '4_right', '5_left', '5_right', '6_left', \
        '6_right', '7_left', '7_right', '8_left', '8_right']

    sensor_data = {}

    for i in range(len(names)):
        str_name = names[i] + '.csv'
        temp = pd.read_csv(str_name))

        #Filter the data
        temp_filt = temp.apply(filter_df, axis=0)

        #Take the Fourier Transform of the data
        temp_FT = temp_filt.apply(np.fft.fft, axis=0)
        temp_FT = temp_FT.apply(np.fft.fftshift(), axis=0).abs()

        #Determine the sampling frequency
        Fs = len(temp)/temp.at[len(temp)-1, 'time'] #samples per second
        dF = Fs/len(temp)
        freq = np.arange(Fs/2, Fs/2, dF))







if __name__=='__main__':
    main()