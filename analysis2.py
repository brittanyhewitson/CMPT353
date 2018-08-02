import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from math import sqrt
from scipy.signal import argrelextrema

OUTPUT_TEMPLATE_CLASSIFIER = (
    'Bayesian classifier: {bayes:.3g}\n'
    'kNN classifier:      {knn:.3g}\n'
    'SVM classifier:      {svm:.3g}\n'
)

OUTPUT_TEMPLATE_REGRESS = (
    'Linear regression:     {lin_reg:.3g}\n'
    'Polynomial regression: {pol_reg:.3g}\n'
)




def filter_df(df):
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, df)
    
# walk_data is the acceleration data to be filtered and transformed.  
# data is the original dataframe read from csv
def filter_and_fft(walk_data, data):
    #Filter the data
    data_filt = walk_data.apply(filter_df, axis=0)
   
    #Take the Fourier Transform of the data
    data_FT = data_filt.apply(np.fft.fft, axis=0)
    data_FT = data_FT.apply(np.fft.fftshift, axis=0)
    data_FT = data_FT.abs()

    #Determine the sampling frequency
    Fs = round(len(data)/data.at[len(data)-1, 'time']) #samples per second
    #dF = Fs/len(temp)
   
    data_FT['freq'] = np.linspace(-Fs/2, Fs/2, num=len(data))
    
    return data_FT

def plot_acc(df, x_axis, output_name):
    plt.figure()
    plt.plot(df[x_axis], df['acceleration'])
    plt.title('Total Linear Acceleration')
    plt.xlabel(x_axis)
    #plt.savefig(output_name + '_acc.png')
    #plt.close()

def plot_vel(df, x_axis, output_name):
    plt.figure()
    plt.plot(df[x_axis], df['velocity'])
    plt.title('Total Angular Velocity')
    plt.xlabel(x_axis)
    plt.savefig(output_name + '_vel.png')
    plt.close()

def eucl_dist_w(df):
    return sqrt(df['wx']**2 + df['wy']**2 + df['wz']**2)

def eucl_dist_a(df):
    return sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    
    
def read_csv(dir, file_ext, n, i):
     
        if dir == 'Data':
            str_name =  dir + '/' + str(i) + file_ext +  '.csv'
        else:
            str_name =  dir + '/' + file_ext + str(i) + '.csv'
			
        return pd.read_csv(str_name)
	
# dir is the file directory, file_ext is the prefix of the file, n is the number of data sets
#3 files should be named like file_ext1, file_ext2, etc.
# mode: ax (acceleration in x), ay (acceleration in y), euclid (euclidian norm of x, y and z)
# extracts the peak of the major spikes in the data 
def analyzePeaks(dir, file_ext, n, mode):
    # extracts the 2 largest peaks that are characteristic of a signal
    important_blips = pd.DataFrame()
    
    for i in range(1,n+1):
	
        # left and right 13 doesnt exist, skip for now
        if i == 13:
            continue            
            
        data = read_csv(dir, file_ext, n, i)
        
        walk_data = pd.DataFrame(columns=['acceleration'])
       
        if mode == 'euclid':
            #Take the Euclidean Norm
            walk_data['acceleration'] = data.apply(eucl_dist_a, axis=1)
        elif mode == ax:
            walk_data['acceleration'] = data[['ax']]
        elif mode == ay:
            walk_data['acceleration'] = data[['ay']]
        else:
            print("error in mode arg for analyzePeaks")
            break;
        
        data_FT = filter_and_fft(walk_data, data)
        
        # ignore low freq noise
        data_FT = data_FT[data_FT['freq'] > 0.4]
        #plot_acc(data_FT[data_FT.acceleration > 50], 'freq', '')
        
        # Get the local max values, keep only the "significant" blips, lets say those above 40% of max blip
        ind = argrelextrema(data_FT.acceleration.values, np.greater)
        local_max = data_FT.acceleration.values[ind]
        local_max = local_max[local_max > 0.5 * local_max.max()]
        important_blips = important_blips.append(data_FT[data_FT['acceleration'].isin(local_max)])

    return important_blips

# dir is the file directory, file_ext is the prefix of the file, n is the number of data sets
#3 files should be named like file_ext1, file_ext2, etc.
# Gets the freq of the peak from both x spikes and y spikes, makes a datapoint with them.
# If theres more than one peak in both x and y, a datapoint is made for each combination
def xy_peak_pairs(dir, file_ext, n):
    # extracts the 2 largest peaks that are characteristic of a signal
    xy_peaks = pd.DataFrame(columns = ['xfreq', 'yfreq'])
    
    for i in range(1,n+1):
	
        # left and right 13 doesn't exist, skip it
        if i == 13:
            continue            
            
        data = read_csv(dir, file_ext, n, i)
        
        walk_data = pd.DataFrame(columns=['ax', 'ay'])       
      
        walk_data[['ax']] = data[['ax']]        
        walk_data[['ay']] = data[['ay']]        
        
        data_FT = filter_and_fft(walk_data, data)
        
        # ignore low freq noise
        data_FT = data_FT[data_FT['freq'] > 0.4]
        #plot_acc(data_FT[data_FT.acceleration > 50], 'freq', '')
        
        # Get the local max values, keep only the "significant" blips, lets say those above 40% of max blip
        indx = argrelextrema(data_FT.ax.values, np.greater)
        indy = argrelextrema(data_FT.ay.values, np.greater)
        
        xlocal_max = data_FT.ax.values[indx]
        xlocal_max = xlocal_max[xlocal_max > 0.5 * xlocal_max.max()]
        ylocal_max = data_FT.ay.values[indy]
        ylocal_max = ylocal_max[ylocal_max > 0.5 * ylocal_max.max()]
        
        xlocal_max_freq = data_FT[data_FT['ax'].isin(xlocal_max)]['freq'].values
        ylocal_max_freq = data_FT[data_FT['ay'].isin(ylocal_max)]['freq'].values
        pairs = np.transpose([np.tile(xlocal_max_freq, len(ylocal_max_freq)), np.repeat(ylocal_max_freq, len(xlocal_max_freq))])
        
        xy_peaks = xy_peaks.append(pd.DataFrame(data=pairs,  columns=['xfreq', 'yfreq']))

    return xy_peaks

def main():

    #right = analyzePeaks('Data/Greyson', 'r', 6, 'euclid')
    #left = analyzePeaks('Data/Greyson', 'l', 6, 'euclid')
    #others_left = analyzePeaks('Data', '_left', 18, 'euclid')
    #others_right = analyzePeaks('Data', '_left', 18, 'euclid')
	
    xy = xy_peak_pairs('Data/Greyson', 'l', 6)
    plt.plot(xy['xfreq'], xy['yfreq'], 'bo')
    plt.title('my x and y central frequencies left')
	
    plt.figure()
    others_xy = xy_peak_pairs('Data', '_left', 18)
    plt.plot(others_xy['xfreq'], others_xy['yfreq'], 'bo')
    plt.title('other people x and y central frequencies left')
    
    plt.figure()
    xy = xy_peak_pairs('Data/Greyson', 'r', 6)
    plt.plot(xy['xfreq'], xy['yfreq'], 'bo')
    plt.title('my x and y central frequencies right')
	
    plt.figure()
    others_xy = xy_peak_pairs('Data', '_right', 18)
    plt.plot(others_xy['xfreq'], others_xy['yfreq'], 'bo')
    plt.title('other people x and y central frequencies right')
    plt.show()
	
	


if __name__=='__main__':
    main()