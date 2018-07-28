import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def filter_df(df):
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, df)

def plot_gForce(df, x_axis):
    plt.subplot(3, 1, 1)
    plt.plot(df[x_axis], df['gFx'])
    plt.subplot(3, 1, 2)
    plt.plot(df[x_axis], df['gFy'])
    plt.subplot(3, 1, 3)
    plt.plot(df[x_axis], df['gFz'])
    plt.show()

def plot_acc(df, x_axis):
    plt.subplot(3, 1, 1)
    plt.plot(df[x_axis], df['ax'])
    plt.subplot(3, 1, 2)
    plt.plot(df[x_axis], df['ay'])
    plt.subplot(3, 1, 3)
    plt.plot(df[x_axis], df['az'])
    #plt.show()

def plot_vel(df, x_axis):
    plt.subplot(3, 1, 1)
    plt.plot(df[x_axis], df['wx'])
    plt.subplot(3, 1, 2)
    plt.plot(df[x_axis], df['wy'])
    plt.subplot(3, 1, 3)
    plt.plot(df[x_axis], df['wz'])
    #plt.show()

def main():
    #data = pd.read_csv('Data_summary.csv')
    names = ['1_left', '1_right', '2_left', '2_right', '4_left', '4_right', '5_left', '5_right', '6_left', \
        '6_right', '7_left', '7_right', '8_left', '8_right']

    sensor_data = {}

    for i in range(len(names)):
        str_name = 'Data/' + names[i] + '.csv'
        temp = pd.read_csv(str_name)

        #Filter the data
        temp_filt = temp.apply(filter_df, axis=0)

        #Take the Fourier Transform of the data
        temp_FT = temp_filt.apply(np.fft.fft, axis=0)
        temp_FT = temp_FT.apply(np.fft.fftshift, axis=0)
        temp_FT = temp_FT.abs()

        #Determine the sampling frequency
        Fs = round(len(temp)/temp.at[len(temp)-1, 'time']) #samples per second
        dF = Fs/len(temp)
        temp_FT['freq'] = np.arange(-Fs/2, Fs/2, dF)

        plot_gForce(temp_FT, 'freq')

        #Store in the dictionary
        str_filt = names[i] + '_filt'
        str_FT = names[i] + '_FT'

        sensor_data[str_filt] = temp_filt
        sensor_data[str_FT] = temp_FT

        del temp

if __name__=='__main__':
    main()