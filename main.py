import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats, interpolate
from math import sqrt, log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn
seaborn.set()

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
    '''
    #Try scaling:
    bayes_model = make_pipeline(
        StandardScaler(),
        GaussianNB()
    )

    knn_model = make_pipeline(
        StandardScaler(), 
        KNeighborsClassifier(n_neighbors=4)
    )

    svc_model = make_pipeline(
        StandardScaler(), 
        SVC(kernel='linear')
    )
    '''

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


def ML_regress(X, y, name_test):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #Linear regression
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X_train, y_train)

    plt.figure()
    plt.plot(X_test, y_test, 'b.')
    plt.plot(X_test, lin_reg.predict(X_test), 'g-')
    plt.xlabel('Step Frequency')
    plt.ylabel(name_test)
    plt.legend(['Original Data', 'Regression Line'])
    plt.title('ML Linear regression for ' + name_test + ' Versus Step Frequency\n Testing Data')
    plt.savefig('ML_regression' + name_test + '.png')
    plt.close()

    plt.figure()
    plt.plot(X_train, y_train, 'b.')
    plt.plot(X_train, lin_reg.predict(X_train), 'g-')
    plt.xlabel('Step Frequency')
    plt.ylabel(name_test)
    plt.legend(['Original Data', 'Regression Line'])
    plt.title('ML Linear regression for ' + name_test + ' Versus Step Frequency\n Training Data')
    plt.savefig('ML_regression' + name_test + '_train.png')
    plt.close()

    #Score is the r-squared value. If this value is negative it means the linear regression is worse than using
    #a horizontal line (i.e. the mean)
    print(OUTPUT_TEMPLATE_ML_REGRESS.format(
        lin_reg=lin_reg.score(X_test, y_test),
    ))


def stats_regress(x, y, name_test):
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    reg = stats.linregress(x_train, y_train)

    plt.figure()
    plt.plot(x_test, y_test, 'b.')
    plt.plot(x_test, x_test*reg.slope + reg.intercept, 'r-', linewidth=3)
    plt.xlabel('Step Frequency')
    plt.ylabel(name_test)
    plt.legend(['Original Data', 'Regression Line'])
    plt.title('Linear regression for ' + name_test + ' Versus Step Frequency\n Testing Data')
    plt.savefig('lin_regression' + name_test + '.png')
    plt.close()

    plt.figure()
    plt.plot(x_train, y_train, 'b.')
    plt.plot(x_train, x_train*reg.slope + reg.intercept, 'r-', linewidth=3)
    plt.xlabel('Step Frequency')
    plt.ylabel(name_test)
    plt.legend(['Original Data', 'Regression Line'])
    plt.title('Linear regression for ' + name_test + ' Versus Step Frequency\n Training Data')
    plt.savefig('lin_regression' + name_test + '_train.png')
    plt.close()

    #Perform polynomial regression
    x_new_test = np.linspace(x_test.min(), x_test.max(), len(x_test))
    x_new_train = np.linspace(x_train.min(), x_train.max(), len(x_train))

    coeff = np.polyfit(x, y, 5)
    y_fit = np.polyval(coeff, x_new_test)
    y_fit_train = np.polyval(coeff, x_new_train)


    plt.figure()
    plt.plot(x_test, y_test, 'b.')
    plt.plot(x_new_test, y_fit, 'go-')
    plt.xlabel('Step Frequency')
    plt.ylabel(name_test)
    plt.legend(['Original Data', 'Regression Curve'])
    plt.title('Polynomial regression for ' + name_test + ' Versus Step Frequency\n Testing Data')
    plt.savefig('poly_regression' + name_test + '.png')
    plt.close()

    plt.figure()
    plt.plot(x_train, y_train, 'b.')
    plt.plot(x_new_train, y_fit_train, 'go-')
    plt.xlabel('Step Frequency')
    plt.ylabel(name_test)
    plt.legend(['Original Data', 'Regression Curve'])
    plt.title('Polynomial regression for ' + name_test + ' Versus Step Frequency\n Training Data')
    plt.savefig('poly_regression' + name_test + '_train.png')
    plt.close()

    #residuals = y_train - (reg.slope*x_train + reg.intercept)
    #plt.figure()
    #plt.plot(x_train, residuals, 'b-')
    #plt.show()

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


def calc_FT(df, temp, i, num):
    #Take the Fourier Transform of the data
    df_FT = df.apply(np.fft.fft, axis=0)
    df_FT = df_FT.apply(np.fft.fftshift, axis=0)
    df_FT = df_FT.abs()

    #Determine the sampling frequency
    Fs = round(len(temp)/(temp['time'].iloc[-1]-temp['time'].iloc[0])) #samples per second
    df_FT['freq'] = np.linspace(-Fs/2, Fs/2, num=len(temp))

    plot_acc(df_FT, 'freq', names[i] + '_' + str(num))
    plot_vel(df_FT, 'freq', names[i] + '_' + str(num))

    #Find the largest peak at a frequency greater than 0 to determine the average steps per second
    temp_FT = df_FT[df_FT.freq > 0.1]
    ind = temp_FT['acceleration'].nlargest(n=1)
    max_ind = ind.idxmax()
    avg_freq = df_FT.at[max_ind, 'freq']

    #Transform the data to fit a normal distribution
    max_val = df_FT['acceleration'].nlargest(n=1)
    max_val_ind = max_val.idxmax()
    df_FT.at[max_val_ind, 'acceleration'] = temp_FT['acceleration'].max()
    #df_FT['acceleration'] = df_FT['acceleration'].apply(log)

    plot_acc(df_FT, 'freq', names[i] + '_transformed_' + str(num))

    return df_FT, avg_freq


def update_spreadsheet(data_sum, names):
    data_sum['freq_1'] = ''
    data_sum['freq_2'] = ''
    data_sum = data_sum.set_index('F_name')
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

        #Apply the Fourier Transform on the data
        data_FT, avg_freq = calc_FT(data_filt, temp, i, 0)
        data_FT_1, avg_freq_1 = calc_FT(data_filt_1, temp_1, i, 1)
        data_FT_2, avg_freq_2 = calc_FT(data_filt_2, temp_2, i, 2)

        #Save the average step frequencies into the main spreadsheet
        data_sum.at[names[i], 'freq_1'] = avg_freq_1
        data_sum.at[names[i], 'freq_2'] = avg_freq_2

        #Save the Fourier Spectrum into a dictionary
        str_filt = names[i] + '_filt'
        str_FT = names[i] + '_FT'

        #sensor_data[str_filt] = data_filt
        sensor_data[str_FT] = data_FT

    return data_sum, sensor_data

def create_visuals(temp_df):
    #Create visualzations
    #Histogram of frequency distribution
    plt.figure()
    temp_freq = temp_df['freq']
    temp_freq.plot.hist(alpha=0.5)
    plt.title('Distribution of Frequencies')
    plt.xlabel('Frequencies (Steps per Second)')
    plt.ylabel('Count')
    plt.savefig('freq_distribution.png')
    #plt.show()
    plt.close()

    #Histogram of weight distribution
    #plt.figure
    f = plt.figure()
    f.add_subplot(1, 3, 1)
    temp_freq = temp_df['Weight']
    temp_freq.plot.hist(alpha=0.5)
    plt.title('Distribution of Weights')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Count')
    #plt.savefig('freq_distribution.png')
    #plt.show()
    #plt.close()

    #Histogram of age distribution
    f.add_subplot(1, 3, 2)
    temp_freq = temp_df['Age']
    temp_freq.plot.hist(alpha=0.5)
    plt.title('Distribution of Ages')
    plt.xlabel('Age')
    plt.ylabel('Count')
    #plt.savefig('freq_distribution.png')
    #plt.show()
    #plt.close()

    #Histogram of height distribution
    f.add_subplot(1, 3, 3)
    temp_freq = temp_df['Height']
    temp_freq.plot.hist(alpha=0.5)
    plt.title('Distribution of Heights')
    plt.xlabel('Height (cm)')
    plt.ylabel('Count')
    plt.savefig('distributions.png')
    #plt.show()
    #plt.close()
    plt.close()

    #Plot the pie charts for what our data consists of
    #Gender breakdown
    plt.figure()
    temp_gender = temp_df.groupby('Gender').aggregate('count')
    temp_gender['Subject'].plot.pie(autopct='%.2f', figsize=(6, 6))
    plt.title('Percentage of Males versus Females')
    plt.savefig('gender_pie.png')
    #plt.show()
    plt.close()

    #Level of Activity breakdown
    plt.figure
    f.add_subplot(1, 3, 2)
    temp_LOA = temp_df.groupby('Level of Activity').aggregate('count')
    temp_LOA['Subject'].plot.pie(autopct='%.2f', figsize=(6, 6))
    plt.title('Breakdown of the Level of Activity of Subjects')
    plt.savefig('LOA_pie.png')
    #plt.show()
    plt.close()

    #Activity of Choice breakdown
    plt.figure()
    f.add_subplot(1, 3, 3)
    temp_AOC = temp_df.groupby('Activity of Choice').aggregate('count')
    temp_AOC['Subject'].plot.pie(autopct='%.2f', figsize=(6, 6))
    plt.title('Breakdown of the Activity of Choice for Subjects')
    plt.savefig('AOC_pie.png')
    #plt.show()
    plt.close()


def main():
    data_sum = pd.read_csv('Data/Data_Summary.csv')

    #Find the average step frequencies for each person's left and right feet
    data_sum, sensor_data = update_spreadsheet(data_sum, names)
    
    #Perform machine learning test on the result
    is_na = pd.isna(data_sum)
    data_sum = data_sum[is_na['Gender'] == False]
    data_sum = data_sum[data_sum['freq_1'] != '']
    data_sum = data_sum[data_sum['freq_2'] != '']

    temp_1 = data_sum.rename(columns={'freq_1': 'freq'})
    temp_2 = data_sum.rename(columns={'freq_2': 'freq'})
    temp_df = pd.concat([temp_1, temp_2], axis=0)
    #temp_df_X = temp_df['freq']#.apply(float)
    X = temp_df[['freq']].values
    
    create_visuals(temp_df)
    
    #See if there is a relationshp between step frequency and level of activity
    print('LEVEL OF ACTIVTY:')
    ML_classifier(X, temp_df['Level of Activity'].values)

    #See if there is a relationship between step frequency and gender
    print('GENDER:')
    ML_classifier(X, temp_df['Gender'].values)

    #See if there is a relationship between step frequency and activity of choice
    print('ACTIVITY OF CHOICE:')
    ML_classifier(X, temp_df['Activity of Choice'].values)

    #Perform a statistical analysis
    #Does each person have a different step frequency
    #Perform a normal test on the data
    '''
    for i in range(len(names_FT)):
        print('Normal test:')
        print(stats.normaltest(sensor_data[names_FT[i]].acceleration).pvalue)
    '''

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

    #Stats Regression
    print('STATS REGRESSION:')
    temp_df = temp_df.sort_values('freq')
    x = temp_df['freq'].apply(float)

    #Perform a stats regression between the height and the frequency
    print('Height')
    stats_regress(x, temp_df['Height'].apply(float), 'Height')

    #Perform a stats regression between the age and the frequency
    print('Age')
    stats_regress(x, temp_df['Age'].apply(float), 'Age')

    #Perform a stats regression between the weight and the frequency
    print('Weight')
    stats_regress(x, temp_df['Weight'].apply(float), 'Weight')

    
    #Perform a machine learning regression
    print('ML REGRESSION:')

    #Find the P-Value to see if there is a relationship between height and step frequency with machine learning
    print('Height')
    ML_regress(X, temp_df['Height'].values, 'Height')

    #Find the P-Value to see if there is a relationship between age and step frequency with machine learning
    print('Age')
    ML_regress(X, temp_df['Age'].values, 'Age')

    #Find the P-Value to see if there is a relationship between weight and step frequency with machine learning
    print('Weight')
    ML_regress(X, temp_df['Weight'].values, 'Weight')

    data_sum.to_csv('output.csv')


if __name__=='__main__':
    main()