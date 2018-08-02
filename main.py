import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats, interpolate
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import math

names = ['1_left', '1_right', '2_left', '2_right', '3_left', '3_right', '4_left', '4_right', '5_left', '5_right', '6_left', \
        '6_right', '7_left', '7_right', '8_left', '8_right', '9_left', '9_right', '10_left', '10_right', \
        '11_left', '11_right', '12_left', '12_right', '13_left', '13_right', '14_left', '14_right', '15_left', '15_right', \
        '16_left', '16_right', '17_left', '17_right', '18_left', '18_right']

names_FT = ['1_left_FT', '1_right_FT', '2_left_FT', '2_right_FT', '3_left_FT', '3_right_FT', '4_left_FT', '4_right_FT', '5_left_FT', '5_right_FT', '6_left_FT', \
        '6_right_FT', '7_left_FT', '7_right_FT', '8_left_FT', '8_right_FT', '9_left_FT', '9_right_FT', '10_left_FT', '10_right_FT', \
        '11_left_FT', '11_right_FT', '12_left_FT', '12_right_FT', '13_left_FT', '13_right_FT', '14_left_FT', '14_right_FT', '15_left_FT', '15_right_FT', \
        '16_left_FT', '16_right_FT', '17_left_FT', '17_right_FT', '18_left_FT', '18_right_FT']

OUTPUT_TEMPLATE_CLASSIFIER = (
    'Bayesian classifier: {bayes:.3g}\n'
    'kNN classifier:      {knn:.3g}\n'
    'SVM classifier:      {svm:.3g}\n'
)

OUTPUT_TEMPLATE_ML_REGRESS = (
    'Linear regression: {lin_reg:.3g}\n'
)

OUTPUT_TEMPLATE_REGRESS = (
    'p-value:       {pval:.3g}\n'
    'r-value:       {rval:.3g}\n'
    'r-squared:     {rsquared:.3g}\n'
    'slope:         {slope:.3g}\n'
    'intercept:     {intercept:.3g}\n'
    'OLS summary:   {summary:}\n'
    'Polynomial coefficients: {pol_reg}\n'
)

def ML_classifier(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    bayes_model = GaussianNB()
    knn_model = KNeighborsClassifier(n_neighbors=3)
    svc_model = SVC(kernel='linear')

    models = [bayes_model, knn_model, svc_model]

    for i, m in enumerate(models):  # yes, you can leave this loop in if you want.
        m.fit(X_train, y_train)
        # plot_predictions(m) # if we create a function to plot the prediction
        # plt.savefig('predictions-%i.png' % (i,))

    print(OUTPUT_TEMPLATE_CLASSIFIER.format(
        bayes=bayes_model.score(X_test, y_test),
        knn=knn_model.score(X_test, y_test),
        svm=svc_model.score(X_test, y_test),
    ))

def ML_regress(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #Linear regression
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X_train, y_train)

    plt.figure()
    plt.plot(X_test, y_test, 'b.')
    plt.plot(X_test, lin_reg.predict(X_test), 'g-')
    plt.savefig('ML_regression.png')

    #Score is the r-squared value. If this value is negative it means the linear regression is worse than using
    #a horizontal line (i.e. the mean)
    print(OUTPUT_TEMPLATE_ML_REGRESS.format(
        lin_reg=lin_reg.score(X_test, y_test),
    ))


def stats_regress(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    reg = stats.linregress(x_train, y_train)

    plt.figure()
    plt.plot(x_test, y_test, 'b.')
    plt.plot(x_test, x_test*reg.slope + reg.intercept, 'r-', linewidth=3)
    plt.savefig('lin_regression.png')
    plt.close()

    x_new_test = np.linspace(x_test.min(), x_test.max(), len(x_test))

    coeff = np.polyfit(x, y, 3)
    y_fit = np.polyval(coeff, x_new_test)

    plt.figure()
    plt.plot(x_test, y_test, 'b.')
    plt.plot(x_new_test, y_fit, 'g-')
    plt.savefig('poly_regression.png')
    plt.close()

    data = pd.DataFrame({'y': y, 'x': x, 'intercept': 1})
    results = sm.OLS(data['y'], data[['x', 'intercept']]).fit()

    print(OUTPUT_TEMPLATE_REGRESS.format(
        pval=reg.pvalue,
        rval=reg.rvalue,
        rsquared=reg.rvalue**2,
        slope=reg.slope,
        intercept=reg.intercept,
        summary=results.summary(),
        pol_reg=coeff,
    ))

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

def update_freq(data_sum, names):
    data_sum['freq_1'] = ''
    data_sum['freq_2'] = ''
    data_sum = data_sum.set_index('F_name')
    #Don't need the dictonary right now
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

        #Split the data in half
        half_val = round(data_filt.shape[0]/2)
        data_filt_1 = data_filt.iloc[:half_val, :]
        data_filt_2 = data_filt.iloc[half_val:, :]

        temp_1 = temp.iloc[:half_val, :]
        temp_2 = temp.iloc[half_val:, :]

        #Take the Fourier Transform of each half of the data
        data_FT_1 = data_filt_1.apply(np.fft.fft, axis=0)
        data_FT_1 = data_FT_1.apply(np.fft.fftshift, axis=0)
        data_FT_1 = data_FT_1.abs()

        data_FT_2 = data_filt_2.apply(np.fft.fft, axis=0)
        data_FT_2 = data_FT_2.apply(np.fft.fftshift, axis=0)
        data_FT_2 = data_FT_2.abs()

        #Determine the sampling frequency
        Fs_1 = round(len(temp_1)/temp_1.at[len(temp_1)-1, 'time']) #samples per second
        Fs_2 = round(len(temp_2)/temp_2.at[(len(temp_2)-1) + len(temp_1), 'time']) #samples per second
        #dF = Fs/len(temp)

        #data_FT['freq'] = np.arange(-Fs/2, Fs/2, dF)
        data_FT_1['freq'] = np.linspace(-Fs_1/2, Fs_1/2, num=len(temp_1))
        data_FT_2['freq'] = np.linspace(-Fs_2/2, Fs_2/2, num=len(temp_2))

        plot_acc(data_FT_1, 'freq', names[i] + '_1')
        plot_vel(data_FT_1, 'freq', names[i] + '_1')

        plot_acc(data_FT_2, 'freq', names[i] + '_2')
        plot_vel(data_FT_2, 'freq', names[i] + '_2')

        #Find the largest peak at a frequency greater than 0 to determine the average steps per second
        temp_FT = data_FT_1[data_FT_1.freq > 0.1]
        ind = temp_FT['acceleration'].nlargest(n=1)
        max_ind = ind.idxmax()
        avg_freq_1 = data_FT_1.at[max_ind, 'freq']

        #Store into the main dataframe and FT dataframe
        data_sum.at[names[i], 'freq_1'] = avg_freq_1

        #Transform the data to fit a normal distribution
        max_val = data_FT_1['acceleration'].nlargest(n=1)
        max_val_ind = max_val.idxmax()
        data_FT_1.at[max_val_ind, 'acceleration'] = temp_FT['acceleration'].max()
        data_FT_1['acceleration'] = data_FT_1['acceleration'].apply(math.log)
        #data_FT['acceleration'] = data_FT['acceleration'].apply(abs)

        plot_acc(data_FT_1, 'freq', names[i] + '_transformed_1')

        temp_FT = data_FT_2[data_FT_2.freq > 0.1]
        ind = temp_FT['acceleration'].nlargest(n=1)
        max_ind = ind.idxmax()
        avg_freq_2 = data_FT_2.at[max_ind, 'freq']

        #Store into the main dataframe and FT dataframe
        data_sum.at[names[i], 'freq_2'] = avg_freq_2

        #Transform the data to fit a normal distribution
        max_val = data_FT_2['acceleration'].nlargest(n=1)
        max_val_ind = max_val.idxmax()
        data_FT_2.at[max_val_ind, 'acceleration'] = temp_FT['acceleration'].max()
        data_FT_2['acceleration'] = data_FT_2['acceleration'].apply(math.log)
        #data_FT['acceleration'] = data_FT['acceleration'].apply(abs)

        plot_acc(data_FT_2, 'freq', names[i] + '_transformed_2')

        #if i==0:
        #    acc_FT = pd.DataFrame({names[i]: [data_FT]})
        #else:
        #    pd.concat([acc_FT, data_FT], axis=1)

        #Store in the dictionary
        str_filt = names[i] + '_filt'
        str_FT = names[i] + '_FT'

        #Don't need the dictionary right now
        sensor_data[str_filt] = data_filt
        sensor_data[str_FT] = data_FT_1

    return data_sum, sensor_data

def main():
    data_sum = pd.read_csv('Data/Data_Summary.csv')

    #Find the average step frequencies for each person's left and right feet
    data_sum, sensor_data = update_freq(data_sum, names)
    
    #Perform machine learning test on the result
    is_na = pd.isna(data_sum)
    data_sum = data_sum[is_na['Gender'] == False]
    data_sum = data_sum[data_sum['freq_1'] != '']
    data_sum = data_sum[data_sum['freq_2'] != '']
    X_1 = data_sum[['freq_1']].values
    X_2 = data_sum[['freq_2']].values
    X = np.concatenate((X_1, X_2), axis=0)
    '''
    X_temp_1 = data_sum.rename(columns={'freq_1': 'freq'})
    X_temp_2 = data_sum.rename(columns={'freq_2': 'freq'})
    X_df = pd.concat([X_temp_1, X_temp_2], axis=0)

    X = X_df[['freq']].values
    '''

    print(X_2)
    
    #See if there is a relationshp between step frequency and level of activity
    print('Level of Activity:')
    y = data_sum['Level of Activity'].values
    y = np.concatenate((y, y), axis=0)
    ML_classifier(X, y)

    #See if there is a relationship between step frequency and gender
    print('Gender:')
    y = data_sum['Gender'].values
    y = np.concatenate((y, y), axis=0)
    ML_classifier(X, y)

    #See if there is a relationship between step frequency and activity of choice
    print('Activity of Choice:')
    y = data_sum['Activity of Choice'].values
    y = np.concatenate((y, y), axis=0)
    ML_classifier(X, y)

    #Perform a statistical analysis
    #Does each person have a different step frequency
    #Perform a normal test on the data
    for i in range(len(names_FT)):
        print('Normal test:')
        print(stats.normaltest(sensor_data[names_FT[i]].acceleration).pvalue)

    #(Use an ANOVA test and note that f p < 0.05 there is a difference between the means of the groups)
    print('ANOVA:')
    anova = stats.f_oneway(sensor_data['1_left_FT'].acceleration, sensor_data['1_right_FT'].acceleration, sensor_data['2_left_FT'].acceleration, sensor_data['2_right_FT'].acceleration,\
    sensor_data['3_left_FT'].acceleration, sensor_data['3_right_FT'].acceleration, sensor_data['4_left_FT'].acceleration, sensor_data['1_right_FT'].acceleration, \
    sensor_data['3_left_FT'].acceleration, sensor_data['3_right_FT'].acceleration, sensor_data['4_left_FT'].acceleration, sensor_data['4_right_FT'].acceleration, \
    sensor_data['5_left_FT'].acceleration, sensor_data['5_right_FT'].acceleration, sensor_data['6_left_FT'].acceleration, sensor_data['6_right_FT'].acceleration, \
    sensor_data['7_left_FT'].acceleration, sensor_data['7_right_FT'].acceleration, sensor_data['8_left_FT'].acceleration, sensor_data['8_right_FT'].acceleration, \
    sensor_data['9_left_FT'].acceleration, sensor_data['9_right_FT'].acceleration, sensor_data['10_left_FT'].acceleration, sensor_data['10_right_FT'].acceleration, \
    sensor_data['11_left_FT'].acceleration, sensor_data['11_right_FT'].acceleration, sensor_data['12_left_FT'].acceleration, sensor_data['12_right_FT'].acceleration, \
    sensor_data['13_left_FT'].acceleration, sensor_data['13_right_FT'].acceleration, sensor_data['14_left_FT'].acceleration, sensor_data['14_right_FT'].acceleration, \
    sensor_data['15_left_FT'].acceleration, sensor_data['15_right_FT'].acceleration, sensor_data['16_left_FT'].acceleration, sensor_data['16_right_FT'].acceleration, \
    sensor_data['17_left_FT'].acceleration, sensor_data['17_right_FT'].acceleration, sensor_data['18_left_FT'].acceleration, sensor_data['18_right_FT'].acceleration)
    #anova = stats.f_oneway(*acc_FT)
    print(anova.pvalue)

    #Perform a stats linear regression between the height and the frequency
    print('Stats Regression:')
    #temp_df_1 = data_sum.sort_values('freq_1')
    #x_1 = temp_df_1['freq_1'].apply(float)
    #y = temp_df_1['Height'].apply(float)
    temp_df_1 = pd.concat([data_sum['Height'], data_sum['freq_1']], axis=1)
    temp_df_1 = temp_df_1.rename(columns={'freq_1':'freq'})
    temp_df_2 = pd.concat([data_sum['Height'], data_sum['freq_2']], axis=1)
    temp_df_2 = temp_df_2.rename(columns={'freq_2':'freq'})
    temp_df = pd.concat([temp_df_1, temp_df_2], axis=0)
    
    temp_df = temp_df.sort_values('freq')
    x = temp_df['freq'].apply(float)
    y = temp_df['Height'].apply(float)

    stats_regress(x, y)

    
    #Find the P-Value to see if there is a relationship between height and step frequency with machine learning
    #print('Regression:')
    ML_regress(temp_df[['freq']].values, data_sum['Height'].values)


    data_sum.to_csv('output.csv')


if __name__=='__main__':
    main()