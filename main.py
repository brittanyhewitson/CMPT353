import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from math import sqrt

def filter_df(df):
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, df)
'''
def plot_gForce(df, x_axis):
    plt.subplot(3, 1, 1)
    plt.plot(df[x_axis], df['gFx'])
    plt.subplot(3, 1, 2)
    plt.plot(df[x_axis], df['gFy'])
    plt.subplot(3, 1, 3)
    plt.plot(df[x_axis], df['gFz'])
    plt.show()
'''

def plot_acc(df, x_axis):
    plt.plot(df[x_axis], df['acceleration'])
    plt.show()

def plot_vel(df, x_axis):
    plt.plot(df[x_axis], df['velocity'])
    plt.show()

def eucl_dist_w(df):
    return sqrt(df['wx']**2 + df['wy']**2 + df['wz']**2)

def eucl_dist_a(df):
    return sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)


def main():
    names = ['1_left', '1_right', '2_left', '2_right', '4_left', '4_right', '5_left', '5_right', '6_left', \
            '6_right', '7_left', '7_right', '8_left', '8_right']

    sensor_data = {}

    for i in range(len(names)):
        str_name =  'Data/' + names[i] + '.csv'
        temp = pd.read_csv(str_name)

        walk_data = pd.DataFrame(columns=['acceleration', 'velocity'])

        #Take the Euclidean Norm
        walk_data['acceleration'] = temp.apply(eucl_dist_a, axis=1)
        walk_data['velocity'] = temp.apply(eucl_dist_w, axis=1)
        

        #Filter the data
        data_filt = walk_data.apply(filter_df, axis=0)

        #Take the Fourier Transform of the data
        data_FT = data_filt.apply(np.fft.fft, axis=0)
        data_FT = data_FT.apply(np.fft.fftshift, axis=0)
        data_FT = data_FT.abs()

        #Determine the sampling frequency
        Fs = round(len(temp)/temp.at[len(temp)-1, 'time']) #samples per second
        dF = Fs/len(temp)
        data_FT['freq'] = np.arange(-Fs/2, Fs/2, dF)

        plot_acc(data_FT, 'freq')
        plot_vel(data_FT, 'freq')

        #Find the largest peak at a frequency greater than 0 to find the average steps per second
        temp_FT = data_FT[data_FT.freq > 0]
        ind = temp_FT['acceleration'].nlargest(n=2)
        max_ind = ind.idxmax()
        avg_freq = data_FT.at[max_ind, 'freq']

        #Store in the dictionary
        str_filt = names[i] + '_filt'
        str_FT = names[i] + '_FT'

        sensor_data[str_filt] = data_filt
        sensor_data[str_FT] = data_FT


if __name__=='__main__':
    main()