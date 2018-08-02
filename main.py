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
    # plt.close('all')

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

    plt.plot(X_train, y_train, 'b.')
    plt.show()
    plt.figure()
    plt.plot(X_test, lin_reg.predict(X_test), 'g.')
    plt.plot(X_train, lin_reg.predict(X_train), 'r-')
    plt.show()

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
    plt.show()

    x_new_test = np.linspace(x_test.min(), x_test.max(), len(x_test))

    coeff = np.polyfit(x, y, 3)
    y_fit = np.polyval(coeff, x_new_test)

    plt.figure()
    plt.plot(x_test, y_test, 'b.')
    plt.plot(x_new_test, y_fit, 'go-')
    plt.show()

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
    data_sum['freq'] = ''
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

        #Take the Fourier Transform of the data
        data_FT = data_filt.apply(np.fft.fft, axis=0)
        data_FT = data_FT.apply(np.fft.fftshift, axis=0)
        data_FT = data_FT.abs()

        #Determine the sampling frequency
        Fs = round(len(temp)/temp.at[len(temp)-1, 'time']) #samples per second
        #dF = Fs/len(temp)

        #data_FT['freq'] = np.arange(-Fs/2, Fs/2, dF)
        data_FT['freq'] = np.linspace(-Fs/2, Fs/2, num=len(temp))

        plot_acc(data_FT, 'freq', names[i])
        plot_vel(data_FT, 'freq', names[i])

        #Find the largest peak at a frequency greater than 0 to determine the average steps per second
        temp_FT = data_FT[data_FT.freq > 0.1]
        ind = temp_FT['acceleration'].nlargest(n=1)
        max_ind = ind.idxmax()
        avg_freq = data_FT.at[max_ind, 'freq']

        #Store into the main dataframe
        data_sum.at[names[i], 'freq'] = avg_freq

        #Store in the dictionary
        str_filt = names[i] + '_filt'
        str_FT = names[i] + '_FT'

        #Don't need the dictionary right now
        sensor_data[str_filt] = data_filt
        sensor_data[str_FT] = data_FT

    return data_sum, sensor_data

def main():
    names = ['1_left', '1_right', '2_left', '2_right', '3_left', '3_right', '4_left', '4_right', '5_left', '5_right', '6_left', \
            '6_right', '7_left', '7_right', '8_left', '8_right', '9_left', '9_right', '10_left', '10_right', \
            '11_left', '11_right', '12_left', '12_right', '14_left', '14_right', '15_left', '15_right', \
            '16_left', '16_right', '17_left', '17_right']
    data_sum = pd.read_csv('Data/Data_Summary.csv')

    #Find the average step frequencies for each person's left and right feet
    data_sum, sensor_data = update_freq(data_sum, names)
    
    #Perform machine learning test on the result
    is_na = pd.isna(data_sum)
    data_sum = data_sum[is_na['Gender'] == False]
    data_sum = data_sum[data_sum['freq'] != '']
    X = data_sum[['freq']].values
    
    #See if there is a relationshp between step frequency and level of activity
    print('Level of Activity:')
    ML_classifier(X, data_sum['Level of Activity'].values)

    #See if there is a relationship between step frequency and gender
    print('Gender:')
    ML_classifier(X, data_sum['Gender'].values)

    #See if there is a relationship between step frequency and activity of choice
    print('Activity of Choice:')
    ML_classifier(X, data_sum['Activity of Choice'].values)

    #Perform a statistical analysis
    #Does each person have a different step frequency
    #(Use an ANOVA test and note that f p < 0.05 there is a difference between the means of the groups)
    print('ANOVA:')
    #anova = stats.f_oneway(sensor_data['1_left_FT'].acceleration, sensor_data['1_right_FT'].acceleration, sensor_data['2_left_FT'].acceleration, sensor_data['2_right_FT'].acceleration)
    #anova = stats.f_oneway(sensor_data)
    #print(anova.pvalue)

    #Perform a stats linear regression between the height and the frequency
    print('Stats Regression:')
    temp_df = data_sum.sort_values('freq')
    x = temp_df['freq'].apply(float)
    y = temp_df['Height'].apply(float)

    stats_regress(x, y)

    
    #Find the P-Value to see if there is a relationship between height and step frequency with machine learning
    #print('Regression:')
    ML_regress(temp_df[['freq']].values, data_sum['Height'].values)

    #Use regression for when input and output are numbers and use classifier for when determining gender, injured, or walking on stairs


    data_sum.to_csv('output.csv')


if __name__=='__main__':
    main()