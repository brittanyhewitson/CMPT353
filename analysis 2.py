import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

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

def plot_acc(df, x_axis, output_name):
    plt.figure()
    plt.plot(df[x_axis], df['acceleration'])
    plt.title('Total Linear Acceleration')
    plt.xlabel(x_axis)
    plt.savefig(output_name + '_acc.png')
    plt.close()

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
	
def analyzePeaks():
    # extracts the 2 largest peaks that are characteristic of a signal
    important_blips = pd.DataFrame()
    
    for i in range(1,7):
        str_name =  'Data/Greyson/r' + str(i) + '.csv'
        left = pd.read_csv(str_name)

        #str_name =  'Data/Greyson/r' + str(i) + '.csv'
        #right = pd.read_csv(str_name)
        
        walk_data = pd.DataFrame(columns=['acceleration'])
       
        #Take the Euclidean Norm
        walk_data['acceleration'] = left.apply(eucl_dist_a, axis=1)
        
        #Filter the data
        data_filt = walk_data.apply(filter_df, axis=0)
       
        #Take the Fourier Transform of the data
        data_FT = data_filt.apply(np.fft.fft, axis=0)
        data_FT = data_FT.apply(np.fft.fftshift, axis=0)
        data_FT = data_FT.abs()

        #Determine the sampling frequency
        Fs = round(len(left)/left.at[len(left)-1, 'time']) #samples per second
        #dF = Fs/len(temp)
       
        data_FT['freq'] = np.linspace(-Fs/2, Fs/2, num=len(left))
        
        # ignore low freq noise
        data_FT = data_FT[data_FT['freq'] > 0.4]
        plot_acc(data_FT[data_FT.acceleration > 50], 'freq', '')
        
        # Get the local max values, keep only the "significant" blips, lets say those above 40% of max blip
        ind = argrelextrema(data_FT.acceleration.values, np.greater)
        local_max = data_FT.acceleration.values[ind]
        local_max = local_max[local_max > 0.5 * local_max.max()]
        important_blips = important_blips.append(data_FT[data_FT['acceleration'].isin(local_max)])

    return important_blips


def main():

    peaks = analyzePeaks()
    plt.plot(peaks['freq'], peaks['acceleration'], 'bo')
    

    data_sum.to_csv('output.csv')


if __name__=='__main__':
    main()